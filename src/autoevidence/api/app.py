"""FastAPI 推理服务。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autoevidence import __version__
from autoevidence.inference.pipeline import Analyzer
from autoevidence.schemas.models import Ticket

app = FastAPI(title="AutoEvidence", version=__version__)
_analyzer: Analyzer | None = None


def get_analyzer() -> Analyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer(mode="demo")
    return _analyzer


class AnalyzeRequest(BaseModel):
    ticket: dict[str, Any]
    request_id: Optional[str] = None
    timeout_s: float = 30.0


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "mode": get_analyzer().mode,
        "ts": time.time(),
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    rid = req.request_id or str(uuid.uuid4())
    try:
        # 拒绝金标进入推理
        payload = dict(req.ticket)
        payload.pop("gold", None)
        ticket = Ticket.model_validate(payload)
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": "validation_error", "msg": str(e), "request_id": rid})

    try:
        t0 = time.time()
        result = get_analyzer().analyze(ticket, request_id=rid)
        if time.time() - t0 > req.timeout_s:
            raise TimeoutError("analyze timeout")
        result["latency_s"] = time.time() - t0
        return result
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail={"error": "timeout", "msg": str(e), "request_id": rid})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "server_error", "msg": str(e), "request_id": rid})


@app.get("/")
def index() -> dict:
    return {
        "service": "AutoEvidence",
        "endpoints": ["/health", "/analyze", "/demo"],
        "note": "轻量展示入口；算法主链优先",
    }


@app.get("/demo")
def demo_page():
    from fastapi.responses import HTMLResponse

    html = """
    <!doctype html>
    <html lang="zh-CN"><head><meta charset="utf-8"/><title>AutoEvidence Demo</title>
    <style>
      body{font-family:ui-sans-serif,system-ui;margin:2rem;max-width:960px}
      textarea{width:100%;height:220px} pre{background:#f6f8fa;padding:1rem;overflow:auto}
      button{padding:.5rem 1rem}
    </style></head>
    <body>
      <h1>AutoEvidence</h1>
      <p>查看输入证据、模型结论、条款对照与分流状态（演示模式）。</p>
      <textarea id="payload">粘贴 Ticket JSON</textarea>
      <p><button onclick="run()">分析</button></p>
      <pre id="out"></pre>
      <script>
      async function run(){
        const ticket = JSON.parse(document.getElementById('payload').value);
        delete ticket.gold;
        const r = await fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({ticket})});
        document.getElementById('out').textContent = JSON.stringify(await r.json(), null, 2);
      }
      </script>
    </body></html>
    """
    return HTMLResponse(html)
