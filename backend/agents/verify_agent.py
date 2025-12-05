import requests
from typing import List, Dict

def verify_links(links: List[str]) -> List[Dict]:
    results = []
    for link in links:
        ok = False
        status = None
        try:
            r = requests.head(link, timeout=3, allow_redirects=True)
            status = r.status_code
            ok = 200 <= r.status_code < 400
        except Exception:
            try:
                r = requests.get(link, timeout=4)
                status = r.status_code
                ok = 200 <= r.status_code < 400
            except Exception:
                ok = False
        results.append({'url': link, 'ok': ok, 'status': status})
    return results
