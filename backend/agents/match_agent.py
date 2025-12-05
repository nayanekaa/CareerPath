from typing import List, Dict

def match_roles(skills: List[str], target: str) -> List[Dict]:
    """Naive matching: score roles from data/roles.csv by overlap with skills or target keywords."""
    import csv, os
    roles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'roles.csv')
    results = []
    roles = []
    try:
        with open(roles_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                roles.append(r)
    except FileNotFoundError:
        # fallback roles
        roles = [
            {'title': 'Data Analyst', 'skills': 'SQL,Python,Excel'},
            {'title': 'Frontend Engineer', 'skills': 'JavaScript,React,HTML,CSS'},
            {'title': 'Backend Engineer', 'skills': 'Python,Node,SQL,Docker'},
        ]

    skillset = set([s.lower() for s in skills])
    for r in roles:
        rskills = set([s.strip().lower() for s in r.get('skills','').split(',') if s.strip()])
        overlap = len(skillset & rskills)
        score = int((overlap / max(1, len(rskills))) * 100)
        # boost if target words match
        if target and target.lower() in r.get('title','').lower():
            score = min(100, score + 20)
        results.append({'role': r.get('title','Unknown'), 'matchScore': score, 'reason': f'{overlap} overlapping skills'})

    results = sorted(results, key=lambda r: r['matchScore'], reverse=True)
    return results
