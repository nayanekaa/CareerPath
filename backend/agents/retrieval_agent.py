import os
import json
from typing import List, Dict

try:
    from sentence_transformers import SentenceTransformer, util
    _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    _EMBED_MODEL = None

_PASSAGES = None

def _load_passages():
    global _PASSAGES
    if _PASSAGES is not None:
        return _PASSAGES
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'roadmaps_corpus.json')
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            _PASSAGES = json.load(f)
    except FileNotFoundError:
        _PASSAGES = []
    return _PASSAGES

def retrieve(query: str, k: int = 4) -> List[Dict]:
    """Retrieve top-k passages. If sentence-transformers available, use embeddings; otherwise simple substring scoring."""
    passages = _load_passages()
    if not passages:
        return []

    if _EMBED_MODEL is not None:
        try:
            texts = [p.get('text','') for p in passages]
            q_emb = _EMBED_MODEL.encode(query, convert_to_tensor=True)
            corpus_embs = _EMBED_MODEL.encode(texts, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, corpus_embs)[0]
            vals, idxs = sims.topk(min(k, len(texts)))
            results = []
            for i,score in zip(idxs, vals):
                i = int(i)
                results.append({'id': passages[i].get('id', i), 'text': passages[i].get('text',''), 'score': float(score)})
            return results
        except Exception:
            pass

    # fallback: simple substring scoring
    scored = []
    q = query.lower()
    for p in passages:
        text = p.get('text','').lower()
        score = text.count(q.split()[0]) if q.split() else 0
        if any(w in text for w in q.split()):
            score += 1
        scored.append((score, p))
    scored = sorted(scored, key=lambda t: t[0], reverse=True)
    results = []
    for s,p in scored[:k]:
        results.append({'id': p.get('id'), 'text': p.get('text'), 'score': s})
    return results
