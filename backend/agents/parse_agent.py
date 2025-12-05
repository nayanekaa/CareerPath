import re
from typing import Dict, List

def parse_resume(text: str) -> Dict:
    """Very lightweight resume parser that extracts skills and a short summary."""
    if not text:
        return {'summary': '', 'skills': []}

    # naive skill extraction: look for common separators & keywords
    skills = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # common skills pattern
        if re.search(r'\b(Python|JavaScript|React|SQL|Docker|Kubernetes|Pandas|Excel|Git|Node)\b', line, re.IGNORECASE):
            for match in re.findall(r'\b(Python|JavaScript|React|SQL|Docker|Kubernetes|Pandas|Excel|Git|Node)\b', line, re.IGNORECASE):
                skills.add(match)

    # fallback: words after 'Skills' header
    m = re.search(r'SKILLS\s*[:\-]?\s*(.*)', text, re.IGNORECASE)
    if m:
        parts = re.split(r'[;,\n]', m.group(1))
        for p in parts:
            p = p.strip()
            if p:
                skills.add(p)

    summary = ' '.join(text.splitlines()[:3])
    return {
        'summary': summary[:1000],
        'skills': list(skills)
    }
