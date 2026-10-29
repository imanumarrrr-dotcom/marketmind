import time, random
from openai import (OpenAI, RateLimitError, APIConnectionError, APITimeoutError,
                    AuthenticationError, BadRequestError, InternalServerError)
from pydantic import ValidationError
from config import PRICES

class LLMError(Exception):
    def __init__(self, kind, msg): super().__init__(msg); self.kind = kind
class BudgetExceeded(Exception): pass

def mask(k): return (k[:3] + "…" + k[-4:]) if k and len(k) > 12 else "••••"

class Usage:
    def __init__(self, rows): self.rows = rows
    def add(self, stage, model, u, lat):
        pi, po = PRICES.get(model, (2.5, 10.0))
        self.rows.append(dict(stage=stage, model=model, tin=u.prompt_tokens, tout=u.completion_tokens,
                              cost=(u.prompt_tokens * pi + u.completion_tokens * po) / 1e6, latency=round(lat, 2)))
    @property
    def cost(self): return sum(r["cost"] for r in self.rows)
    @property
    def tokens(self): return sum(r["tin"] + r["tout"] for r in self.rows)

class LLM:
    """The single place every API call passes: timeout, bounded backoff+jitter, distinct error kinds, cost accounting."""
    def __init__(self, key, cfg, usage, log):
        self.c = OpenAI(api_key=key, timeout=cfg.timeout, max_retries=0)
        self.cfg, self.usage, self.log = cfg, usage, log
        comp = self.c.chat.completions
        self._parse = getattr(comp, "parse", None) or self.c.beta.chat.completions.parse
        self._create = comp.create

    def _call(self, stage, fn, **kw):
        cfg = self.cfg; model = cfg.models[stage]
        if not model.startswith(("o1", "o3", "o4", "gpt-5")): kw["temperature"] = cfg.temps[stage]
        for i in range(cfg.max_retries + 1):
            if self.usage.cost >= cfg.budget_usd: raise BudgetExceeded(f"cost ceiling ${cfg.budget_usd:.2f} reached")
            t = time.time()
            try:
                r = fn(model=model, **kw); self.usage.add(stage, model, r.usage, time.time() - t); return r
            except AuthenticationError as e: raise LLMError("auth", "Authentication failed - check the API key") from e
            except BadRequestError as e:
                raise LLMError("context_length" if getattr(e, "code", "") == "context_length_exceeded" else "bad_request", str(e)[:300]) from e
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
                kind = "rate_limit" if isinstance(e, RateLimitError) else "timeout" if isinstance(e, APITimeoutError) else "transient"
                if i == cfg.max_retries: raise LLMError(kind, f"{kind}: gave up after {i + 1} attempts") from e
                try: d = float(e.response.headers.get("retry-after"))
                except Exception: d = min(30, 2 ** i) + random.random()
                self.log("retry", f"{stage}: {kind} - attempt {i + 1}, backing off {d:.1f}s"); time.sleep(d)

    def chat(self, stage, messages, tools=None):
        kw = dict(messages=messages)
        if tools: kw["tools"] = tools
        m = self._call(stage, self._create, **kw).choices[0].message
        if getattr(m, "refusal", None): raise LLMError("refusal", m.refusal)
        if not m.content and not m.tool_calls: raise LLMError("empty", "empty response")
        return m

    def structured(self, stage, messages, cls):
        """API-level schema constraint + independent pydantic validation; one retry with the error fed back."""
        msgs, err = list(messages), ""
        for _ in range(2):
            try:
                m = self._call(stage, self._parse, messages=msgs, response_format=cls).choices[0].message
                if getattr(m, "refusal", None): raise LLMError("refusal", m.refusal)
                if m.parsed is None: raise ValueError("no parsed content")
                return cls.model_validate(m.parsed.model_dump())
            except (ValidationError, ValueError) as e:
                err = str(e)[:400]; self.log("defect", f"{stage}: structured output invalid: {err[:120]}")
                msgs = msgs + [{"role": "user", "content": f"Your previous output failed validation: {err}. Return a corrected object."}]
        raise LLMError("validation", err)
