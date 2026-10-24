"""LAN base-station receiver for SwarmMind store-and-forward reports (no cloud).

    python scripts/uplink_sink.py --port 8800 --out data/basestation/reports.jsonl

POST /reports   store a report (idempotent by job_id; duplicates return {"duplicate": true})
GET  /reports   all received reports as JSON
GET  /          HTML table of received reports
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse


def load_existing(path: Path) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    if not path.is_file():
        return reports
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("job_id"):
                reports.setdefault(str(rec["job_id"]), rec)
    return reports


def _cell(value) -> str:
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return html.escape("" if value is None else str(value))


def _fmt_ts(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return str(ts or "")


def create_app(out_path: Path) -> FastAPI:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reports = load_existing(out_path)
    app = FastAPI(title="SwarmMind base station")

    @app.post("/reports")
    async def post_report(request: Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid JSON")
        if not isinstance(body, dict) or not body.get("job_id"):
            raise HTTPException(status_code=400, detail="report must be an object with job_id")
        jid = str(body["job_id"])
        if jid in reports:
            return JSONResponse({"ok": True, "duplicate": True, "job_id": jid})
        rec = dict(body)
        rec["job_id"] = jid
        rec["received_ts"] = time.time()
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        reports[jid] = rec
        answer = str(rec.get("answer", "")).replace("\n", " ")
        print(f"[basestation] report {jid} nodes={rec.get('nodes_used')} "
              f"handoffs={len(rec.get('handoffs') or [])}\n  Q: {rec.get('question', '')}\n"
              f"  A: {answer[:300]}", flush=True)
        return JSONResponse({"ok": True, "duplicate": False, "job_id": jid})

    @app.get("/reports")
    async def get_reports():
        return JSONResponse(list(reports.values()))

    @app.get("/", response_class=HTMLResponse)
    async def index():
        rows = []
        for rec in reversed(list(reports.values())):
            rows.append(
                "<tr>"
                f"<td>{_cell(_fmt_ts(rec.get('received_ts')))}</td>"
                f"<td>{_cell(rec.get('job_id'))}</td>"
                f"<td>{_cell(rec.get('question'))}</td>"
                f"<td class='ans'>{_cell(rec.get('answer'))}</td>"
                f"<td>{_cell(', '.join(map(str, rec.get('nodes_used') or [])))}</td>"
                f"<td>{len(rec.get('handoffs') or [])}</td>"
                "</tr>")
        body = "\n".join(rows) or "<tr><td colspan='6'>No reports received yet.</td></tr>"
        page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5"><title>Base Station Reports</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #444;padding:6px;vertical-align:top;text-align:left}}
th{{background:#222}} td.ans{{white-space:pre-wrap;max-width:50em}}
</style></head><body>
<h1>SwarmMind base station</h1><p>{len(reports)} report(s) received. Auto-refreshes every 5 s.</p>
<table><thead><tr><th>Received</th><th>Job</th><th>Question</th><th>Answer</th><th>Nodes</th><th>Handoffs</th></tr></thead>
<tbody>
{body}
</tbody></table></body></html>"""
        return HTMLResponse(page)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="SwarmMind LAN base-station report sink")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--out", default="data/basestation/reports.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    app = create_app(out)
    print(f"[basestation] listening on {args.host}:{args.port}, storing to {out.resolve()}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
