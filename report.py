import time
from .schemas import Report
from .state import now
SECS = ["market_overview", "key_trends", "opportunities", "risks", "recommendations"]
TITLES = dict(market_overview="Market overview", key_trends="Key trends", opportunities="Opportunities", risks="Risks", recommendations="Recommendations")

def build_report(st, corpus, cfg):
    docs = {d["id"]: d for d in corpus}; syn = st.synthesis; ev = st.evidence
    cited = {i for s in SECS for f in syn.get(s, []) for i in f["evidence_ids"]}
    cited |= {i for m in st.matrix.values() for cell in m.values() for i in cell["evidence_ids"]}
    cited |= {i for i, e in ev.items() if e["claim_type"] == "uncertainty"}
    app = [ev[i] for i in sorted(cited) if i in ev]
    lim = []
    if st.partial_reason: lim.append(f"RUN INCOMPLETE ({st.partial_reason}): findings cover only the questions answered below.")
    for _, q in st.iter_q():
        if st.q_status.get(q["id"]) != "answered": lim.append(f"[{q['id']}] {st.q_status.get(q['id'], 'open').upper()}: {q['question']}")
    lim += [f"[{e['evidence_id']}] {e['claim']}" for e in ev.values() if e["claim_type"] == "uncertainty"]
    for ent, m in st.matrix.items():
        lim += [f"Matrix: {ent} / {a} not established in the corpus" for a, c in m.items() if c["status"] == "not_established" and a in ("pricing_entry", "pricing_enterprise")]
    lim += [f"[QC {d['id']}] unresolved {d['type']} ({d['severity']}): {d['description']}" for d in st.qc.get("defects", []) if not d.get("declared") and st.qc.get("verdict") == "fail"]
    n = len(st.q_status) or 1; ans = sum(v == "answered" for v in st.q_status.values()); cov = ans / n
    spec = [e for e in ev.values() if e["claim_type"] != "uncertainty"]
    strong = sum(e["credibility"] == "high" for e in spec) / max(1, len(spec)); conflicts = sum("conflict" in e["analyst_notes"] for e in ev.values())
    lvl = "high" if cov >= .9 and not conflicts and st.qc.get("verdict") == "pass" and not st.partial_reason and strong >= .3 else "medium" if cov >= .6 else "low"
    srcs = {}
    for e in app:
        d = docs.get(e.get("doc_id"))
        if d: srcs[d["id"]] = dict(doc_id=d["id"], title=d["title"], source=d["source"], date=d["date"], credibility=d["credibility"])
    st_models = {k: v for k, v in cfg.models.items()}
    rep = dict(report_id=f"MM-{st.run_id}", generated_at=now(), model_versions=st_models, run_cost=round(sum(r["cost"] for r in st.usage), 4),
               partial=bool(st.partial_reason), partial_reason=st.partial_reason,
               research_objective=dict(request=st.request, scope={k: st.scope.get(k) for k in ("entity_type", "segment", "geography", "time_horizon", "deliverable")},
                                       assumptions=[a["assumption"] for a in st.scope.get("ambiguities", [])], ambiguities=[a["issue"] for a in st.scope.get("ambiguities", [])]),
               executive_summary=syn.get("executive_summary", ""), competitor_analysis=dict(matrix=st.matrix),
               evidence_appendix=app, confidence_level=lvl,
               confidence_rationale=f"coverage {ans}/{n} questions answered; {sum(e['credibility']=='high' for e in spec)}/{len(spec)} records from high-credibility sources; {conflicts} unresolved conflicts; QC verdict {st.qc.get('verdict','n/a')}.",
               limitations_and_gaps=lim, sources=list(srcs.values()),
               metadata=dict(tokens=sum(r["tin"] + r["tout"] for r in st.usage), tool_calls=st.tool_calls, iterations=st.iteration,
                             elapsed_s=round(time.time() - st.started, 1), repair_rounds=st.repair_rounds, flags=len(st.flags)),
               approval=None, **{s: syn.get(s, []) for s in SECS})
    return rep

def validate_report(rep, st):
    """Schema validation plus cross-field invariants that JSON Schema cannot express."""
    errs = []
    try: Report.model_validate(rep)
    except Exception as e: return [f"schema: {str(e)[:200]}"]
    for s in SECS:
        for f in rep[s]:
            for i in f["evidence_ids"]:
                if i not in st.evidence: errs.append(f"{s}: evidence id {i} does not resolve")
            if f["claim_type"] == "fact" and not f["evidence_ids"]: errs.append(f"{s}: fact without evidence: {f['statement'][:50]}")
    lim = " ".join(rep["limitations_and_gaps"])
    for _, q in st.iter_q():
        if st.q_status.get(q["id"]) != "answered" and f"[{q['id']}]" not in lim: errs.append(f"question {q['id']} missing from limitations")
    if not rep["limitations_and_gaps"] and not all(v == "answered" for v in st.q_status.values()): errs.append("limitations empty but questions unanswered")
    return errs

def chip(t, html): return f'<span class="chip {t}">{t.upper()}</span>' if html else f"**[{t.upper()}]**"

def render_md(r, html=False):
    o = ["# Market intelligence report", f"`{r['report_id']}` · generated {r['generated_at']} · confidence **{r['confidence_level'].upper()}**"]
    if r["partial"]: o.append(f"> ⚠️ **INCOMPLETE RUN: {r['partial_reason']}**")
    ro = r["research_objective"]
    o += ["## Research objective", ro["request"], "**Scope:** " + "; ".join(f"{k}: {v}" for k, v in ro["scope"].items()),
          "**Recorded assumptions:**"] + [f"- {a}" for a in ro["assumptions"]] + ["## Executive summary", r["executive_summary"] or "_not generated_"]
    for s in SECS:
        o.append(f"## {TITLES[s]}")
        o += [f"- {chip(f['claim_type'], html)} {f['statement']} `[{', '.join(f['evidence_ids'])}]` (confidence: {f['confidence']})" for f in r[s]] or ["_No supported findings._"]
    o.append("## Competitor matrix")
    m = r["competitor_analysis"]["matrix"]
    if m:
        at = list(next(iter(m.values())))
        o += ["| Entity | " + " | ".join(at) + " |", "|" + "---|" * (len(at) + 1)]
        for e, row in m.items():
            o.append(f"| {e} | " + " | ".join(("⚠ conflict" if row[a]["status"] == "conflict" else "—" if row[a]["status"] == "not_established" else "✔ " + ",".join(row[a]["evidence_ids"])) for a in at) + " |")
        o.append("_— = not established in the corpus; ⚠ = sources conflict._")
    o += ["## Limitations and gaps"] + [f"- {x}" for x in r["limitations_and_gaps"]] or ["- none"]
    o += ["## Confidence", r["confidence_rationale"], "## Evidence appendix", "| ID | Type | Claim | Source | Cred. | Conf. |", "|---|---|---|---|---|---|"]
    o += [f"| {e['evidence_id']} | {chip(e['claim_type'], html)} | {e['claim']} | {e['source_detail']} | {e['credibility']} | {e['confidence']} |" for e in r["evidence_appendix"]]
    md = r["metadata"]
    o += ["## Run metadata", f"Models: {r['model_versions']} · tokens {md['tokens']} · cost ${r['run_cost']} · tool calls {md['tool_calls']} · elapsed {md['elapsed_s']}s"]
    a = r.get("approval"); o.append("**Approval:** " + (f"{a['decision']} by {a['approver']} at {a['timestamp']}. {a['notes']}" if a else "pending human review"))
    return "\n\n".join(o)
