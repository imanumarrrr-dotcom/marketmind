import json, os, time, uuid
from dataclasses import dataclass, field, asdict

def now(): return time.strftime("%Y-%m-%dT%H:%M:%S")

@dataclass
class State:
    request: str = ""
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4])
    scope: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    q_status: dict = field(default_factory=dict)      # qid -> open | answered | unanswerable
    evidence: dict = field(default_factory=dict)
    rejections: list = field(default_factory=list)
    results: dict = field(default_factory=dict)       # result_id -> tool result (provenance)
    tool_log: list = field(default_factory=list)
    log: list = field(default_factory=list)
    usage: list = field(default_factory=list)
    iteration: int = 0
    tool_calls: int = 0
    tool_counts: dict = field(default_factory=dict)
    stage: str = "new"
    status: str = "new"           # new | running | awaiting_approval | published | rejected
    partial_reason: str = ""
    defects: list = field(default_factory=list)
    qc: dict = field(default_factory=dict)
    synthesis: dict = field(default_factory=dict)
    matrix: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    approval: dict = field(default_factory=dict)
    repair_rounds: int = 0
    chaos: dict = field(default_factory=dict)
    flags: list = field(default_factory=list)         # injection flags
    audit: list = field(default_factory=list)
    terminations: list = field(default_factory=list)
    last_sig: str = ""
    strikes: int = 0
    started: float = field(default_factory=time.time)
    elapsed: float = 0.0

    def __post_init__(self): self._hook = None

    def add_log(self, kind, msg):
        e = dict(t=now(), kind=kind, msg=str(msg)[:400]); self.log.append(e)
        if self._hook: self._hook(e)

    def iter_q(self):
        for o in self.plan.get("objectives", []):
            for q in o["sub_questions"]: yield o, q

    def save(self, root="runs"):
        self.elapsed = round(time.time() - self.started, 1)
        d = os.path.join(root, self.run_id); os.makedirs(d, exist_ok=True)
        def w(n, o):
            with open(os.path.join(d, n), "w") as f: json.dump(o, f, indent=2, default=str)
        w("state.json", asdict(self)); w("plan.json", self.plan); w("tool_log.json", self.tool_log)
        w("evidence.json", dict(evidence=self.evidence, rejections=self.rejections)); w("qc.json", self.qc)
        w("report.json", self.report); w("usage.json", self.usage); w("audit.json", self.audit)
        return d

    @classmethod
    def load(cls, path):
        with open(path) as f: return cls(**json.load(f))

def _flags_docs(self): return {f["doc"] for f in self.flags if f.get("doc")}
State.flags_docs = _flags_docs
