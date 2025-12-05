import os, json
try:
    import openai
    OPENAI_PRESENT = True
except Exception:
    OPENAI_PRESENT = False

DEMO_PATH = os.path.join(os.path.dirname(__file__), '..', 'demo', 'canned_outputs.json')

def gen_synthetic_resumes(seed_profile: str, n: int = 10):
    if OPENAI_PRESENT and os.environ.get('OPENAI_API_KEY'):
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        prompt = f"Generate {n} anonymized 2-sentence resume summaries based on this profile: {seed_profile}. Return JSON array of objects with fields: summary, skills (comma separated), projects (array)."
        try:
            resp = openai.ChatCompletion.create(
                model=os.environ.get('OPENAI_MODEL','gpt-4o-mini'),
                messages=[{"role":"user","content":prompt}],
                temperature=0.8,
                max_tokens=800
            )
            txt = resp['choices'][0]['message']['content']
            # try parse JSON
            try:
                data = json.loads(txt)
                return data
            except Exception:
                return [{'summary': txt[:200], 'skills': seed_profile, 'projects': []}]
        except Exception:
            pass

    # fallback: generate simple templated resumes
    out = []
    for i in range(n):
        out.append({
            'summary': f'{seed_profile} candidate variant #{i+1}. Strong in {seed_profile.split()[0]}.',
            'skills': seed_profile,
            'projects': [f'Project {i+1} - Demo task', 'Capstone project']
        })
    return out
