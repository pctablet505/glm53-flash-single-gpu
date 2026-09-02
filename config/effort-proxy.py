"""Reasoning-effort control for GLM-5.3-Flash, exposed as model aliases.

Why this exists
---------------
GLM-5.3's chat template resolves reasoning effort like this:

    effective_reasoning_effort = reasoning_effort
        if reasoning_effort in ['low', 'high'] else 'max'

so a request that does not set it gets **max** -- unbounded thinking. In
practice that means the model emits an unterminated <think> block, burns the
whole token budget, and returns an EMPTY assistant message. It reads as the
model refusing or ignoring you; it is neither.

The parameter can only be passed via ``chat_template_kwargs``, which neither
VS Code's BYOK "customendpoint" provider nor Codex lets you set. Both, however,
let you pick a **model**. So this proxy publishes one alias per effort level and
injects the parameter on the way through:

    glm53-flash-low   -> reasoning_effort=low    fast, terse thinking
    glm53-flash-high  -> reasoning_effort=high   deliberate
    glm53-flash       -> unset = 'max'       deepest reasoning (default)

`max` is the template's own default and gives the best answers, but it thinks
without a bound: on some prompts it fills the whole token budget with an
unterminated <think> block and returns content=null (the thinking is still
there, under the `reasoning` field -- clients just do not render it). This proxy
detects that and transparently retries at `high`, so callers always get an
answer.
"""

from __future__ import annotations

import json
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM = os.environ.get("GLM_UPSTREAM", "http://127.0.0.1:8002")
UPSTREAM_MODEL = os.environ.get("GLM_UPSTREAM_MODEL", "glm53-flash")

# alias -> reasoning_effort. 'max' is offered but flagged: it is the upstream
# default and the one that produces empty replies without a large budget.
ALIASES = {
    "glm53-flash": None,      # default = max (template's own default)
    "glm53-flash-low": "low",
    "glm53-flash-high": "high",
    "glm53-flash-max": None,  # pass through untouched = template's 'max'
}

app = FastAPI(title="glm53-effort-proxy")
client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=10.0))


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "owned_by": "local",
                    "root": UPSTREAM_MODEL,
                    "max_model_len": 262144,
                }
                for name in ALIASES
            ],
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    try:
        r = await client.get(f"{UPSTREAM}/v1/models", timeout=5.0)
        return JSONResponse({"ok": r.status_code == 200, "upstream": UPSTREAM})
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


def _promote_reasoning(data: dict) -> dict | None:
    """Move ``reasoning`` into ``content`` when there is no answer text.

    Last resort, used only when a retry also came back answerless. A blank
    assistant turn is not neutral -- it poisons the next request's history --
    so surfacing the model's own reasoning is strictly better than nothing.
    """
    try:
        msg = data["choices"][0]["message"]
        if (msg.get("content") or "").strip() or msg.get("tool_calls"):
            return None
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        if not reasoning:
            return None
        msg["content"] = reasoning
        return data
    except Exception:
        return None


def _lost_to_truncation(data: dict) -> bool:
    """True if the reply is USELESS to a client: no content and no tool call.

    Two ways GLM-5.3 produces one:

    * ``finish_reason="length"`` -- the template ends the prompt with a bare
      ``<think>``, so the model starts inside a reasoning block. At
      ``reasoning_effort=max`` it thinks unbounded and can exhaust the budget
      before emitting ``</think>``.
    * ``finish_reason="stop"``   -- it finishes thinking and stops without
      writing an answer.

    Either way ``content`` is null. The thinking itself survives in the
    ``reasoning`` field, but no client renders that, so the user sees a blank
    turn -- and worse, the blank assistant turn goes back into the history on
    the next request. GLM's template renders it as an empty assistant message,
    the model reads the thread as fresh, and replies "I'm ready to help. What
    would you like me to do?" That is the context-loss symptom; the context was
    never lost, the transcript was poisoned by our own empty turns.

    A message with tool_calls and null content is NORMAL and must not be
    retried -- that is how every tool call is returned.
    """
    try:
        ch = data["choices"][0]
        msg = ch.get("message") or {}
        if msg.get("tool_calls"):
            return False
        return not (msg.get("content") or "").strip()
    except Exception:
        return False


# --- request tracing (GLM_TRACE=1) -------------------------------------------
# Turned on to diagnose agent "context loss": it records, per request, how much
# history the client actually sent and what came back. Off by default; writes one
# JSON line per turn, no message text, so it is safe to leave enabled.
TRACE_PATH = os.environ.get("GLM_TRACE_PATH", "/tmp/glm-proxy-trace.jsonl")


def _trace(event: dict) -> None:
    if os.environ.get("GLM_TRACE") != "1":
        return
    try:
        event["t"] = time.strftime("%H:%M:%S")
        with open(TRACE_PATH, "a") as fh:
            fh.write(json.dumps(event) + "\n")
    except Exception:
        pass


def _summarize_request(body: dict) -> dict:
    msgs = body.get("messages") or []
    roles: dict[str, int] = {}
    blank_assistant = 0
    for m in msgs:
        r = m.get("role", "?")
        roles[r] = roles.get(r, 0) + 1
        if r == "assistant" and not (m.get("content") or "").strip() and not m.get("tool_calls"):
            blank_assistant += 1
    chars = sum(len(m.get("content") or "") for m in msgs if isinstance(m.get("content"), str))
    return {
        "alias": body.get("model"),
        "n_messages": len(msgs),
        "roles": roles,
        # A blank assistant turn is the poison described in _lost_to_truncation.
        "blank_assistant_turns": blank_assistant,
        "history_chars": chars,
        "n_tools": len(body.get("tools") or []),
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "stream": bool(body.get("stream")),
    }


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def forward(request: Request):
    body = await request.json()
    alias = body.get("model", "")
    effort = ALIASES.get(alias)   # unknown alias -> pass through (max)

    _trace({"dir": "req", **_summarize_request(body)})
    body["model"] = UPSTREAM_MODEL
    if effort is not None:
        kw = dict(body.get("chat_template_kwargs") or {})
        # An explicit caller-supplied value wins; the alias only fills a gap.
        kw.setdefault("reasoning_effort", effort)
        body["chat_template_kwargs"] = kw

    url = f"{UPSTREAM}{request.url.path}"
    if body.get("stream"):
        async def stream():
            saw_text = False
            buffered: list[bytes] = []

            def _has_text(raw: bytes) -> bool:
                """True if an SSE chunk carries a NON-EMPTY delta.

                A substring test on b'"content":"' is not enough: the stream is
                full of '"content":""' and '"content":null' frames, which would
                falsely look like output and defeat the fallback.
                """
                for line in raw.split(b"\n"):
                    line = line.strip()
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload == b"[DONE]":
                        continue
                    try:
                        d = json.loads(payload)
                    except Exception:
                        continue
                    for ch in d.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if (delta.get("content") or "").strip():
                            return True
                        # A tool call is a perfectly good answer with no text.
                        if delta.get("tool_calls"):
                            return True
                return False

            async with client.stream("POST", url, json=body) as r:
                async for chunk in r.aiter_raw():
                    if saw_text:
                        yield chunk
                        continue
                    buffered.append(chunk)
                    if _has_text(chunk):
                        saw_text = True
                        for b in buffered:
                            yield b
                        buffered.clear()
            if saw_text:
                return
            # No answer text at all: the model spent the budget thinking and
            # never closed </think>. Redo with bounded thinking so the caller
            # gets an actual answer rather than an empty message.
            retry = dict(body)
            kw = dict(retry.get("chat_template_kwargs") or {})
            kw["reasoning_effort"] = "high"
            retry["chat_template_kwargs"] = kw
            # Non-streaming retry on purpose: it lets the SAME guard the
            # non-streaming path uses run here too. Streaming the retry meant an
            # answerless retry reached the client as another blank turn -- the
            # one outcome we must never produce, because the blank goes back
            # into history and the model then reads the thread as fresh (see
            # _lost_to_truncation). This path is rare; losing incremental
            # output on it costs nothing.
            retry["stream"] = False
            r2 = await client.post(url, json=retry)
            data2 = json.loads(r2.text)
            if _lost_to_truncation(data2):
                data2 = _promote_reasoning(data2) or data2
            msg = ((data2.get("choices") or [{}])[0].get("message")) or {}
            delta: dict = {"role": "assistant"}
            if msg.get("content"):
                delta["content"] = msg["content"]
            if msg.get("tool_calls"):
                delta["tool_calls"] = [
                    {**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])
                ]
            if len(delta) == 1:
                delta["content"] = (
                    "[no answer produced: the model exhausted its token budget "
                    "while reasoning. Retry, or use the glm53-flash-high alias "
                    "for bounded thinking.]"
                )
            head = {
                "id": data2.get("id", "chatcmpl-retry"),
                "object": "chat.completion.chunk",
                "created": data2.get("created", 0),
                "model": data2.get("model", UPSTREAM_MODEL),
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            tail = dict(head)
            tail["choices"] = [{
                "index": 0,
                "delta": {},
                "finish_reason": (data2.get("choices") or [{}])[0].get(
                    "finish_reason", "stop"),
            }]
            yield ("data: " + json.dumps(head) + "\n\n").encode()
            yield ("data: " + json.dumps(tail) + "\n\n").encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    r = await client.post(url, json=body)
    data = json.loads(r.text)

    if _lost_to_truncation(data):
        # The model thought past its budget and produced no answer. Retry with
        # bounded thinking. See _lost_to_truncation.
        retry = dict(body)
        kw = dict(retry.get("chat_template_kwargs") or {})
        kw["reasoning_effort"] = "high"
        retry["chat_template_kwargs"] = kw
        r2 = await client.post(url, json=retry)
        data2 = json.loads(r2.text)
        if not _lost_to_truncation(data2):
            return JSONResponse(data2, status_code=r2.status_code)
        # Both attempts produced an answerless turn. Returning it blank is the
        # worst option: the client renders nothing AND sends the empty turn
        # back as history, which makes the model read the thread as fresh.
        # Promote the reasoning into content so the transcript stays coherent.
        data = _promote_reasoning(data2) or _promote_reasoning(data) or data

    return JSONResponse(data, status_code=r.status_code)
