"""Utility to build an embeddings index for the roadmaps corpus.
This is a lightweight script that will use sentence-transformers to create and save embeddings.
"""
import os, json
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'roadmaps_corpus.json')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus_embeddings.json')

def build():
    if SentenceTransformer is None:
        print('sentence-transformers not installed; skipping index build')
        return
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    texts = [p.get('text','') for p in corpus]
    embs = model.encode(texts)
    out = [{'id': p.get('id'), 'text': p.get('text'), 'embedding': e.tolist()} for p,e in zip(corpus, embs)]
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f)
    print('index written to', OUT_PATH)

if __name__ == '__main__':
    build()
