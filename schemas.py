from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict

class S(BaseModel): model_config = ConfigDict(extra="forbid")
Conf = Literal["high", "medium", "low"]
CT = Literal["fact", "inference", "recommendation", "uncertainty"]
EvType = Literal["pricing", "features", "market_size", "competitor_profile", "regulatory", "trend"]
ToolName = Literal["search_information", "retrieve_document", "calculate_metric"]

class Ambiguity(S):
    issue: str; handling: Literal["clarify", "assume"]; assumption: str; question: str
class Scope(S):
    entity_type: str; segment: str; geography: str; time_horizon: str; deliverable: str
    ambiguities: List[Ambiguity]; needs_clarification: bool
class SubQ(S):
    question: str; evidence_type: EvType; tools: List[ToolName]; priority: Literal["high", "medium", "low"]
class Objective(S):
    title: str; sub_questions: List[SubQ]
class Plan(S):
    objectives: List[Objective]
class Finding(S):
    statement: str; claim_type: CT; evidence_ids: List[str]; confidence: Conf
class Synthesis(S):
    market_overview: List[Finding]; key_trends: List[Finding]; opportunities: List[Finding]
    risks: List[Finding]; recommendations: List[Finding]
class ExecSummary(S): text: str
class Contradiction(S):
    evidence_ids: List[str]; description: str; gap_question: str
class Contradictions(S): items: List[Contradiction]

class Report(BaseModel):            # final report contract (Section 6.1), validated in code
    model_config = ConfigDict(extra="forbid")
    report_id: str; generated_at: str; model_versions: dict; run_cost: float
    partial: bool; partial_reason: str
    research_objective: dict
    executive_summary: str
    market_overview: List[Finding]; key_trends: List[Finding]
    competitor_analysis: dict
    opportunities: List[Finding]; risks: List[Finding]; recommendations: List[Finding]
    evidence_appendix: List[dict]
    confidence_level: Conf; confidence_rationale: str
    limitations_and_gaps: List[str]
    sources: List[dict]
    metadata: dict
    approval: Optional[dict] = None

def export_schemas(d):
    import json, os; os.makedirs(d, exist_ok=True)
    for n, m in dict(plan=Plan, scope=Scope, report=Report, synthesis=Synthesis).items():
        json.dump(m.model_json_schema(), open(f"{d}/{n}.schema.json", "w"), indent=2)
