"""CPU fixtures only: never invoke the CUDA worker or real train.main."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import run_final_batch_gd_cuda as runner
from tools.gd_forensic_protocol import planned_gpu_matrix


def fixture_dump(layout='final_9'):
    slots = runner._slots(layout)
    d = {
        'image_id': [runner.FINAL_IDS[i] for i in slots],
        'pairs': [torch.tensor([[j, j+1] for j in range(runner.FINAL_PAIR_COUNTS[i])]) for i in slots],
        'pred_vocab': [f'p{i:02}' for i in range(50)] + ['relation'],
        'background_predicate_indices': [50],
        'predicate_alias_map': {'alias': 'p00'},
        'pred_emb': torch.zeros(51, 768),
    }
    for field in ('model_logits', 'prior_rows', 'text_logits', 'cls_logits', 'adaptive_logits'):
        d[field] = [torch.arange(51).float().repeat(runner.FINAL_PAIR_COUNTS[i], 1) for i in slots]
    for field in ('subj_label', 'obj_label', 'gt_subj_label', 'gt_obj_label'):
        d[field] = [['object']*runner.FINAL_PAIR_COUNTS[i] for i in slots]
    d['gt_subj_idx'] = [p[:, 0].tolist() for p in d['pairs']]
    d['gt_obj_idx'] = [p[:, 1].tolist() for p in d['pairs']]
    d['gt_pred'] = [[f'p{j%2:02}' for j in range(len(p))] for p in d['pairs']]
    return d


@pytest.mark.parametrize('layout', ['final_9', 'final_9_padded_to_12'])
def test_trim_only_image_fields_preserves_full_vocabulary_and_65_rows(layout):
    source = fixture_dump(layout)
    result = runner._trim_dump(source, layout)
    assert result['n_images'] == 9 and result['n_pairs'] == 65
    assert len(result['pred_vocab']) == 51
    assert result['background_predicate_indices'] == [50]
    assert result['pred_emb'].shape == (51, 768)
    assert result['predicate_alias_map'] == source['predicate_alias_map']
    assert sum(len(p) for p in result['pairs']) == 65
    assert len(source['image_id']) == len(runner._slots(layout))


def test_reject_first_nine_or_wrong_pair_counts():
    d = fixture_dump()
    d['image_id'][0] = 'first_validation_image'
    with pytest.raises(ValueError, match='identity'):
        runner._trim_dump(d, 'final_9')
    with pytest.raises(ValueError, match='65'):
        runner._assert_population(runner.FINAL_IDS, [1]*9, 'final_9')


@pytest.mark.parametrize('layout', ['final_9', 'final_9_padded_to_12'])
def test_dataset_selects_original_indices_and_deliberate_duplicates(monkeypatch, layout):
    import openvocab_rel.datasets.vg150_loader as vg
    class FakeDataset:
        def __init__(self, split):
            self.split = split
            self.cfg = SimpleNamespace(samples_per_epoch=10000)
            self.rows = [{'image_id': 'prefix'} for _ in range(10392)] + [
                {'image_id': iid, 'source_index': 10392+i} for i, iid in enumerate(runner.FINAL_IDS)]
    monkeypatch.setattr(vg, 'VG150JSONLDataset', FakeDataset)
    restore = runner._final_dataset_patch(layout)
    try:
        d = FakeDataset('validation')
        assert [r['source_index'] for r in d.rows] == [10392+i for i in runner._slots(layout)]
        assert len(FakeDataset('train').rows) == 10401
        assert d.cfg.samples_per_epoch == 0
    finally:
        restore()


@pytest.mark.parametrize('value', [torch.tensor(1.), torch.tensor(4), torch.ones(2, dtype=torch.bfloat16)])
def test_hash_supports_scalar_buffers_and_bfloat16(value):
    assert runner._tensor_hash(value) == runner._tensor_hash(value.clone())


def test_hash_fingerprints_buffers_plain_mutable_state_and_trainability():
    m = torch.nn.Linear(2, 2).eval()
    m.register_buffer('counter', torch.tensor(0))
    m.cache = {'last': torch.ones(2)}
    before = runner._object_hashes(m)
    m.counter.add_(1)
    m.cache['last'][0] = 9
    m.weight.requires_grad_(False)
    after = runner._object_hashes(m)
    assert before['buffers'] != after['buffers']
    assert before['plain_module_state'] != after['plain_module_state']
    assert before['parameters']['weight']['requires_grad'] is True
    assert after['parameters']['weight']['requires_grad'] is False


def test_full_numpy_rng_hash_detects_middle_element_change():
    old = np.random.get_state()
    try:
        before = runner._rng_state(False)
        state = np.random.get_state()
        state[1][300] ^= np.uint32(1)
        np.random.set_state(state)
        after = runner._rng_state(False)
        assert before['numpy_sha256'] != after['numpy_sha256']
        assert len(before['numpy_state'][1]) == 624
        assert before['torch_cpu_state_hex']
        assert before['python_state']
    finally:
        np.random.set_state(old)


def test_checkpoint_p_installed_exactly_after_registration(tmp_path):
    from openvocab_rel.models.relational_model import RelationalModel
    model = torch.nn.Module()
    E, P = torch.zeros(51, 768), torch.ones(51, 768)
    path = tmp_path / 'neutral_filename.pt'
    torch.save({'model': {'predicate_prototypes': P}}, path)
    saved = runner._checkpoint_p(path)
    RelationalModel.init_readout_v2(model, E)
    assert torch.equal(model.predicate_prototypes, E)
    runner._install_checkpoint_p(model, saved)
    assert runner._tensor_hash(model.predicate_prototypes) == runner._tensor_hash(P)
    assert model.predicate_prototypes.requires_grad
    assert torch.equal(model._readout_v2_anchor, E)
    with pytest.raises(ValueError, match='requires'):
        runner._install_checkpoint_p(model, None)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predicate_prototypes = torch.nn.Parameter(torch.ones(2, 2))
        self.register_buffer('counter', torch.tensor(0))
    def adaptive_predicate_logits(self, rel):
        return rel @ self.predicate_prototypes


def test_instrument_event_order_state_capture_and_one_forward_bound():
    m, clip = ToyModel().eval(), torch.nn.Linear(2, 2).eval()
    inst = runner.Instrument('final_9')
    cfg = SimpleNamespace(amp=False, amp_dtype='bf16')
    def forward(*args, **kw):
        m.counter.add_(1)
        torch.rand(1)
        return None, [torch.ones(1, 2)], None
    try:
        with torch.no_grad():
            inst.forward(forward, cfg, m, clip, None, [], torch.device('cpu'))
            m.adaptive_predicate_logits(torch.ones(1, 2))
        assert inst.before['model']['buffers'] != inst.after['model']['buffers']
        assert inst.before['rng'] != inst.after['rng']
        assert inst.events[-1]['before_rel_feat_returned'] is False
        assert not any(inst.before['model_modes'].values())
        assert inst.runtime['grad_enabled'] is False
        with pytest.raises(RuntimeError, match='one relational forward'):
            inst.forward(forward, cfg, m, clip, None, [], torch.device('cpu'))
    finally:
        inst.restore()


def test_direct_upstream_prototype_read_is_blocked():
    m, clip = ToyModel().eval(), torch.nn.Identity().eval()
    inst = runner.Instrument('final_9')
    def bad(*args, **kw):
        return None, [m.predicate_prototypes], None
    try:
        with pytest.raises(RuntimeError, match='before rel_feat'):
            inst.forward(bad, SimpleNamespace(amp=False), m, clip, None, [], torch.device('cpu'))
        assert inst.events[-1]['event'] == 'upstream_prototype_attribute_read'
    finally:
        inst.restore()


@pytest.mark.parametrize('layout', ['final_9', 'final_9_padded_to_12'])
def test_pixel_hashes_shape_and_padding_are_recorded(layout):
    slots = runner._slots(layout)
    batch = [{'image_id': runner.FINAL_IDS[i]} for i in slots]
    pixels = torch.stack([torch.full((3, 2, 2), float(i)) for i in slots])
    counts = [runner.FINAL_PAIR_COUNTS[i] for i in slots]
    prep = lambda *a: (pixels, [torch.ones(2, 4) for _ in slots], [[(0, 1)]*n for n in counts])
    inst = runner.Instrument(layout)
    inst.prepare(prep, None, batch, None, torch.device('cpu'))
    assert inst.input['pixel_shape'][0] == len(slots)
    assert len(inst.input['per_image_pixel_sha256']) == len(slots)
    assert inst.input['historical_pixel_identity'] == 'UNPROVEN_FROM_HISTORICAL_DUMPS'
    assert inst.input['source_indices'][:9] == list(range(10392, 10401))
    if len(slots) == 12:
        assert inst.input['duplicate_slots'] == [{'slot': 9+i, 'duplicates_slot': i} for i in range(3)]


def test_repeat_launches_separate_interpreters_with_identical_environment(tmp_path, monkeypatch):
    calls = []
    def fake_run(cmd, **kw):
        calls.append((cmd, kw['env']))
        cond = cmd[-1]
        (tmp_path / cond).mkdir()
        torch.save({'condition': cond}, tmp_path / cond / 'record.pt')
    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    plans = planned_gpu_matrix()
    for p in plans[:2]:
        runner._launch_condition(p, SimpleNamespace(out=tmp_path/'out.json'), tmp_path)
    assert len(calls) == 2
    assert calls[0][0][:-1] == calls[1][0][:-1]
    assert calls[0][1] == calls[1][1]
    assert calls[0][1]['PYTHONHASHSEED'] == '1234'


def test_pairwise_comparisons_use_retained_tensors_and_detect_unstable_repeat():
    def record(name, x):
        return dict(condition=name, rel_feat_tensor=x, input={}, before={}, runtime={})
    records = [record('X_R0_noP_9', torch.zeros(65, 768)),
               record('X_R0_noP_9_repeat', torch.ones(65, 768))]
    pairwise = runner._comparisons(records)
    assert len(pairwise) == 1
    assert records[1]['same_condition_repeatability']['changed_row_count'] == 65
    assert records[1]['same_condition_repeatability']['max_abs'] == 1
    assert all(records[1]['repeat_controls_match'].values())


def test_local_wprd_excludes_padding_and_preserves_51_column_normalization(tmp_path, monkeypatch):
    root = tmp_path/'data'
    root.mkdir()
    pv = fixture_dump()['pred_vocab']
    (root/'frequency_prior_train.json').write_text(json.dumps({'predicate_vocab': pv, 'global_log_probs': [-1.0]*51}))
    monkeypatch.setattr(runner, 'DATA_ROOT', root)
    outputs = []
    for layout in ('final_9', 'final_9_padded_to_12'):
        d = runner._trim_dump(fixture_dump(layout), layout)
        path = tmp_path/(layout+'.pt')
        torch.save(d, path)
        outputs.append(runner._smoke_wprd(path, tmp_path))
    assert outputs[0] == outputs[1]
    assert outputs[0]['n_gt_rows'] == 65
    assert outputs[0]['text']['wprd_macro'] == 0.5
    assert outputs[0]['full_endpoint_inertness_established'] is False


def test_output_refuses_overwrite_and_unknown_gpu_state(tmp_path):
    path = tmp_path/'evidence.json'
    runner._write_json(path, {'status': 'GD-UNRESOLVED'})
    with pytest.raises(FileExistsError):
        runner._write_json(path, {})
    p = dict(nvidia_smi={'returncode': 0}, compute_query_ok=False,
             compute_processes=[], torch_cuda_available=True, torch_cuda_device_count=1)
    assert not runner._gpu_ready(p)
    p['compute_query_ok'] = True
    assert runner._gpu_ready(p)
    p['compute_processes'] = ['123, other_project, 4000']
    assert not runner._gpu_ready(p)


def test_constructed_arguments_for_all_nine_conditions_without_model_launch(tmp_path, monkeypatch):
    from openvocab_rel import train
    captured = {}
    class StopBeforeConstruction(Exception):
        pass
    def fake_main(argv):
        parsed = dict(zip(argv[::2], argv[1::2]))
        captured[parsed['--run_name'].removeprefix('paper_c_gd_')] = parsed
        raise StopBeforeConstruction()
    monkeypatch.setattr(train, 'main', fake_main)
    monkeypatch.setattr(runner, '_checkpoint_p', lambda p: torch.ones(51, 768) if p.name.startswith('readout') else None)
    for plan in planned_gpu_matrix():
        with pytest.raises(StopBeforeConstruction):
            runner._run_one(plan, SimpleNamespace(), tmp_path)
    for name, argv in captured.items():
        assert argv['--eval_fast_mode'] == 'false'
        assert argv['--epochs'] == '0' and argv['--eval_batches'] == '1'
        assert argv['--batch_size'] == '12' and argv['--seed'] == '1234'
        assert argv['--readout_v2_enabled'] == ('true' if name.startswith(('Z_', 'Zp_')) else 'false')
        assert '--reset_epoch' not in argv
    for original in ('X_R0_noP_9', 'Z_R2_P_registered_9'):
        def computation_args(argv):
            ignored = {'--run_name', '--out_dir', '--save_metrics_json', '--eval_sgg_dump_pair_logits_path'}
            return {k: v for k, v in argv.items() if k not in ignored}
        assert computation_args(captured[original]) == computation_args(captured[original+'_repeat'])


def test_repeat_path_exclusion_does_not_hide_computation_changes():
    a = {'cfg': {'out_dir': 'a', 'amp': True}, 'parameter_hash': 'abc'}
    b = {'cfg': {'out_dir': 'b', 'amp': True}, 'parameter_hash': 'abc'}
    assert runner._without_output_paths(a) == runner._without_output_paths(b)
    b['cfg']['amp'] = False
    assert runner._without_output_paths(a) != runner._without_output_paths(b)


def test_parent_report_never_automatically_promotes_synthetic_matrix(tmp_path, monkeypatch):
    out, work = tmp_path/'report.json', tmp_path/'work'
    monkeypatch.setattr(runner.sys, 'argv', ['runner', '--out', str(out), '--work-dir', str(work)])
    monkeypatch.setattr(runner, '_preflight', lambda: dict(nvidia_smi={'returncode': 0},
        compute_query_ok=True, compute_processes=[], torch_cuda_available=True, torch_cuda_device_count=1))
    # Explicitly replace process launching; this test never probes/allocates CUDA.
    def fixture_condition(plan, *a):
        return dict(condition=plan['condition'], rel_feat_tensor=torch.zeros(65, 768),
                    input={}, before={}, runtime={})
    monkeypatch.setattr(runner, '_launch_condition', fixture_condition)
    assert runner.main() == 0
    record = json.loads(out.read_text())
    assert len(record['conditions']) == 9
    assert len(record['pairwise_rel_feat']) == 36
    assert record['status'] == 'GD-UNRESOLVED'
    assert record['automatic_promotion_enabled'] is False
    assert record['decision_flags_are_unassessed_placeholders'] is True
    assert record['human_interpretation_required'] is True
