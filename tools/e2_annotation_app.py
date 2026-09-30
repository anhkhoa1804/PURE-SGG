"""Small local, filesystem-backed E2 annotation application.

Each process serves one annotator only. It never loads another annotator's
answers and writes only to annotations/<annotator_id>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ANNOTATORS = {"annotator_1", "annotator_2", "annotator_3"}
TASK_VERSION = "paper-c-e2-development-v1"


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def state_payload(packet: Path, annotator: str, annotation_file: Path) -> dict:
    """Build the browser state for exactly one annotator.

    Keeping this as a pure function makes the isolation rule testable without
    opening a network socket in CI.
    """
    image_rows = _read_jsonl(packet / "human_annotation_tasks.jsonl")
    block_rows = _read_jsonl(packet / "human_block_tasks.jsonl")
    image_rows = [row for row in image_rows if row["annotator_id"] == annotator]
    block_rows = [row for row in block_rows if row["annotator_id"] == annotator]
    saved = {}
    if annotation_file.exists():
        for row in _read_jsonl(annotation_file):
            saved[f"{row['kind'].replace('_truth','').replace('block_solvability','block')}:{row['task_key']}"] = row
    def image_task(row):
        allowed = {
            "annotator_id", "candidate_id", "block_side", "image_id",
            "subject_label", "object_label", "relation_a", "relation_b",
        }
        x = {key: row[key] for key in allowed}
        x["task_key"] = f"{row['candidate_id']}:{row['block_side']}"
        x["image_url"] = f"/media/images/{row['image_id']}.jpg"
        x["overlay_url"] = f"/media/overlays/{row['image_id']}.jpg"
        return x
    def block_task(row):
        allowed = {
            "annotator_id", "candidate_id", "image_a", "image_b",
            "subject_label", "object_label", "relation_a", "relation_b",
            "description_a", "description_b",
        }
        x = {key: row[key] for key in allowed}
        x["task_key"] = row["candidate_id"]
        x["image_a_url"] = f"/media/overlays/{row['image_a']}.jpg"
        x["image_b_url"] = f"/media/overlays/{row['image_b']}.jpg"
        return x
    return {"annotator_id": annotator, "task_version": TASK_VERSION,
            "image_tasks": [image_task(x) for x in image_rows],
            "block_tasks": [block_task(x) for x in block_rows], "saved": saved}


def _html() -> str:
    return r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Paper C E2 annotation</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1200px;margin:20px auto;padding:0 18px;color:#222}
header{display:flex;justify-content:space-between;align-items:center;gap:20px}
.bar{background:#f2f2f2;padding:10px;border-radius:6px;margin:10px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{border:1px solid #ccc;border-radius:8px;padding:12px;background:#fff}
img{max-width:100%;max-height:460px;object-fit:contain;background:#eee}
label{display:block;margin:7px 0} select,input,textarea,button{font:inherit;padding:6px}
textarea{width:100%;min-height:60px;box-sizing:border-box}
button{cursor:pointer;margin-right:8px}.hidden{display:none}.error{color:#a00}.ok{color:#075}
table{border-collapse:collapse}td,th{padding:5px 9px;border-bottom:1px solid #ddd;text-align:left}
</style></head><body>
<header><h1>Paper C E2 annotation</h1><div id="who"></div></header>
<div class="bar"><b>Independent session.</b> This interface shows only this annotator's tasks and answers. It does not show model scores or other annotations.<br>
<span id="progress"></span></div>
<div id="message"></div>
<div id="task"></div>
<script>
let state=null, current=0, kind='image';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, opts){let r=await fetch(path,opts);let j=await r.json();if(!r.ok)throw new Error(j.error||'request failed');return j;}
function sel(name, values, value){return `<select id="${name}">${values.map(v=>`<option value="${esc(v)}" ${value===v?'selected':''}>${esc(v)}</option>`).join('')}</select>`}
function recordFor(k){return state.saved[k]||{};}
function render(){
 const arr=kind==='image'?state.image_tasks:state.block_tasks;if(!arr.length){document.getElementById('task').innerHTML='<p>No tasks.</p>';return}
 current=Math.max(0,Math.min(current,arr.length-1)); const t=arr[current]; const key=kind+':'+t.task_key; const r=recordFor(key);
 document.getElementById('progress').textContent=`${kind==='image'?'Image truth':'2×2 solvability'} task ${current+1}/${arr.length}`;
 if(kind==='image'){
  document.getElementById('task').innerHTML=`<div class="card"><h2>${esc(t.block_side)} — ${esc(t.subject_label)} → ${esc(t.object_label)}</h2>
   <div class="grid"><div><p>Original image</p><img src="${t.image_url}"></div><div><p>Highlighted image: red S, blue O</p><img src="${t.overlay_url}"></div></div>
   <p>Assess the ordered relation from subject S to object O using visible image evidence only.</p>
   <label>Subject visible ${sel('subject_visible',['yes','no','uncertain'],r.subject_visible||'')}</label>
   <label>Object visible ${sel('object_visible',['yes','no','uncertain'],r.object_visible||'')}</label>
   <label>Relation A: <b>${esc(t.relation_a)}</b> ${sel('truth_a',['valid','invalid','uncertain','not_visible'],r.truth_a||'')}</label>
   <label>Relation B: <b>${esc(t.relation_b)}</b> ${sel('truth_b',['valid','invalid','uncertain','not_visible'],r.truth_b||'')}</label>
   <label>Pair truth ${sel('pair_truth',['relation_a_only','relation_b_only','both_valid','neither_visible','uncertain'],r.pair_truth||'')}</label>
   <label>Direction A ${sel('direction_a',['correct','reversed','uncertain'],r.direction_a||'')}</label>
   <label>Direction B ${sel('direction_b',['correct','reversed','uncertain'],r.direction_b||'')}</label>
   <label>Confidence ${sel('confidence_1_to_5',['1','2','3','4','5'],String(r.confidence_1_to_5||''))}</label>
   <label>Notes <textarea id="notes">${esc(r.notes||'')}</textarea></label>
   <button onclick="saveImage()">Save answer</button><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button></div>`;
 } else {
  document.getElementById('task').innerHTML=`<div class="card"><h2>Block ${current+1}</h2><p>Choose which description matches each image. Use only the image and highlighted roles.</p>
   <div class="grid"><div><p>Image A</p><img src="${t.image_a_url}"><p>${esc(t.description_a)}<br>${esc(t.description_b)}</p></div><div><p>Image B</p><img src="${t.image_b_url}"><p>${esc(t.description_a)}<br>${esc(t.description_b)}</p></div></div>
   <label>Image A choice ${sel('image_a_choice',['description_a','description_b','uncertain'],r.image_a_choice||'')}</label>
   <label>Image B choice ${sel('image_b_choice',['description_a','description_b','uncertain'],r.image_b_choice||'')}</label>
   <label>Confidence ${sel('confidence_1_to_5',['1','2','3','4','5'],String(r.confidence_1_to_5||''))}</label>
   <label>Notes <textarea id="notes">${esc(r.notes||'')}</textarea></label>
   <button onclick="saveBlock()">Save answer</button><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button></div>`;
 }
}
function move(d){current+=d;render()}
function val(id){return document.getElementById(id).value}
async function saveImage(){
 const t=state.image_tasks[current];const answer={kind:'image_truth',task_key:t.task_key,candidate_id:t.candidate_id,block_side:t.block_side,image_id:t.image_id,
  subject_visible:val('subject_visible'),object_visible:val('object_visible'),truth_a:val('truth_a'),truth_b:val('truth_b'),pair_truth:val('pair_truth'),direction_a:val('direction_a'),direction_b:val('direction_b'),confidence_1_to_5:Number(val('confidence_1_to_5')),notes:document.getElementById('notes').value}; await save(answer);
}
async function saveBlock(){const t=state.block_tasks[current];const answer={kind:'block_solvability',task_key:t.task_key,candidate_id:t.candidate_id,image_a:t.image_a,image_b:t.image_b,image_a_choice:val('image_a_choice'),image_b_choice:val('image_b_choice'),confidence_1_to_5:Number(val('confidence_1_to_5')),notes:document.getElementById('notes').value};await save(answer)}
async function save(answer){try{await api('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(answer)});state.saved[(kind==='image'?'image:':'block:')+answer.task_key]=answer;document.getElementById('message').innerHTML='<p class="ok">Saved.</p>';render()}catch(e){document.getElementById('message').innerHTML='<p class="error">'+esc(e.message)+'</p>'}}
async function init(){state=await api('/api/state');document.getElementById('who').textContent=state.annotator_id;render()}
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1)});init();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    server_version = "PaperC-E2-Annotation/1.0"

    @property
    def cfg(self):
        return self.server.cfg  # type: ignore[attr-defined]

    def _json(self, payload: dict, status=HTTPStatus.OK):
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, content_type: str):
        if not path.is_file() or self.cfg.packet not in path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = _html().encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == "/api/state":
            self._json(state_payload(self.cfg.packet, self.cfg.annotator, self.cfg.annotation_file))
            return
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "media" and parts[1] in {"images", "overlays"}:
            name = Path(unquote(parts[2])).name
            self._file(self.cfg.packet / parts[1] / name, "image/jpeg")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            row = json.loads(self.rfile.read(length))
            if row.get("annotator_id", self.cfg.annotator) != self.cfg.annotator:
                raise ValueError("annotator mismatch")
            if row.get("kind") not in {"image_truth", "block_solvability"}:
                raise ValueError("invalid task kind")
            row["annotator_id"] = self.cfg.annotator
            row["task_version"] = TASK_VERSION
            row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            with self.cfg.lock:
                existing = [x for x in _read_jsonl(self.cfg.annotation_file) if x["task_key"] != row["task_key"] or x["kind"] != row["kind"]] if self.cfg.annotation_file.exists() else []
                self.cfg.annotation_file.parent.mkdir(parents=True, exist_ok=True)
                self.cfg.annotation_file.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in existing + [row]), encoding="utf-8")
            self._json({"status": "saved", "task_key": row["task_key"]})
        except Exception as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, *_args):
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--annotator", choices=sorted(ANNOTATORS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    args.packet = args.packet.resolve()
    args.annotation_file = args.packet / "annotations" / f"{args.annotator}.jsonl"
    args.lock = threading.Lock()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.cfg = args  # type: ignore[attr-defined]
    print(f"Paper C E2 annotation app: http://{args.host}:{args.port}/", flush=True)
    print(f"annotator={args.annotator} output={args.annotation_file}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
