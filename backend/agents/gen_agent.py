import os
from typing import Tuple, Dict, Any
import json

DEMO_PATH = os.path.join(os.path.dirname(__file__), '..', 'demo', 'canned_outputs.json')


def _call_gemini(prompt: str, temperature: float = 0.3) -> str:
    """Attempt to call Gemini via the google.generativeai Python package if available and GEMINI_API_KEY is set."""
    try:
        import google.generativeai as genai
    except Exception:
        return None

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = os.environ.get('GEMINI_MODEL', 'text-bison-001')
        # prefer simple generate_text if available
        if hasattr(genai, 'generate_text'):
            resp = genai.generate_text(model=model, prompt=prompt, temperature=temperature)
            # different wrapper versions return text in different fields
            if hasattr(resp, 'text'):
                return resp.text
            if isinstance(resp, dict):
                # try common fields
                if 'candidates' in resp and resp['candidates']:
                    return resp['candidates'][0].get('content') or resp['candidates'][0].get('text')
                if 'output' in resp and isinstance(resp['output'], str):
                    return resp['output']
        else:
            # fallback to a generic generate call
            resp = genai.generate(model=model, prompt=prompt, temperature=temperature)
            if isinstance(resp, dict):
                for k in ('candidates','outputs'):
                    if k in resp and resp[k]:
                        first = resp[k][0]
                        if isinstance(first, dict):
                            return first.get('content') or first.get('text') or str(first)
                        return str(first)
        return None
    except Exception:
        return None


def _call_openai(prompt: str, temperature: float = 0.3) -> str:
    try:
        import openai
    except Exception:
        return None

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None

    try:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=800
        )
        return resp['choices'][0]['message']['content']
    except Exception:
        return None


def _call_llm(prompt: str, temperature: float = 0.3) -> str:
    # 1) Try Gemini
    gemini_resp = _call_gemini(prompt, temperature=temperature)
    if gemini_resp:
        return gemini_resp

    # 2) Try OpenAI
    openai_resp = _call_openai(prompt, temperature=temperature)
    if openai_resp:
        return openai_resp

    # 3) Demo fallback
    if os.path.exists(DEMO_PATH):
        with open(DEMO_PATH, 'r', encoding='utf-8') as f:
            canned = json.load(f)
        return canned.get('analysis', {}).get('final') or canned.get('analysis', {}).get('draft') or 'Demo response'

    return 'Demo response: no LLM key available.'


def generate_roadmap_chain(parsed: Dict, target_role: str, weekly_hours: int, retrieved: list, psychometric: dict) -> Tuple[str, str, str]:
    """Simple draft -> critique -> refine chain using Gemini (preferred), OpenAI, or demo outputs."""
    candidate_summary = parsed.get('summary', '')
    ctx = '\n\n'.join([f"[{r.get('id')}] {r.get('text')[:500]}" for r in retrieved])

    draft_prompt = (
        f"Context:\n{ctx}\n\nCandidate: {candidate_summary}\nTarget Role: {target_role}\nWeekly Hours: {weekly_hours}\nPsychometric: {psychometric}\n\n"
        "Produce a concise 12-week roadmap split into 3 phases. Cite context ids next to major recommendations. Output as plain text."
    )
    draft = _call_llm(draft_prompt, temperature=0.2)

    critique_prompt = f"Critique the following draft for clarity, contradictions and hallucinations. Mark unsupported claims:\n\n{draft}"
    critique = _call_llm(critique_prompt, temperature=0.2)

    refine_prompt = (
        f"Refine the draft using the critique. Produce the final roadmap and avoid adding unsupported claims.\n\n"
        f"Draft:\n{draft}\n\nCritique:\n{critique}"
    )
    final = _call_llm(refine_prompt, temperature=0.2)

    return draft or '', critique or '', final or ''
