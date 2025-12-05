from typing import List, Dict
from ..agents.retrieval_agent import retrieve

def build_rag_prompt(candidate_summary: str, target_role: str, retrieved: List[Dict]) -> str:
    ctx = "\n\n".join([f"[{r['id']}] {r['text']}" for r in retrieved])
    prompt = f"""
Context (retrieved documents):
{ctx}

Task:
Given candidate: {candidate_summary}
Target role: {target_role}

Using only the context above as supporting evidence, produce a 12-week roadmap split into Weeks 1-4 / 5-8 / 9-12. Cite the context id(s) next to recommendations.
"""
    return prompt
