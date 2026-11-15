"""Niveau 1 : RAG sur les recettes Wikilivres.

Pipeline : découpage -> index BM25 + index d'embeddings -> fusion des deux classements (RRF)
-> les meilleurs extraits sont donnés au LLM, qui répond en citant [n] ou s'abstient.

Réglages (variables d'environnement, utiles pour comparer des variantes dans le rapport) :
    RAG_MODE         hybrid (défaut) | dense | bm25
    RAG_CHUNK_CHARS  taille cible d'un extrait en caractères (défaut 800)
    RAG_TOP_K        nombre d'extraits donnés au LLM (défaut 4)
    RAG_MIN_SIM      si > 0 : en dessous de cette similarité cosinus (mode dense/hybrid),
                     on s'abstient sans appeler le LLM (défaut 0 = désactivé)
    CORPUS_DIR       dossier des documents (défaut data/corpus)

Les embeddings sont calculés une fois puis mis en cache dans data/index/ (ignoré par git).
Limite connue : les tokens d'embedding ne sont pas comptés dans « tokens/requête ».
"""
import hashlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src import llm

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", ROOT / "data" / "corpus"))
INDEX_DIR = ROOT / "data" / "index"
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "800"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0"))
MODE = os.getenv("RAG_MODE", "hybrid")
N_CANDIDATES = 10  # extraits gardés après fusion (sert aussi à Recall@5 / MRR au niveau document)
RRF_K = 60

ABSTAIN = "Je ne trouve pas la réponse dans les documents."

SYSTEM = (
    "Tu es un chef anti-gaspi. Réponds en français, UNIQUEMENT à partir des extraits de recettes "
    "fournis, sans rien inventer. Cite la source de chaque affirmation par son numéro entre crochets, "
    "par exemple [1]. Si les extraits ne contiennent pas la réponse, réponds exactement : "
    f"« {ABSTAIN} » Si la question est trop vague pour être traitée, pose une question de précision."
)

STOPWORDS = set(
    "le la les un une des de du d l et ou a au aux en dans pour par sur avec sans que qui quoi est "
    "sont ce cet cette ces il elle on je tu nous vous ne pas plus se sa son ses leur y mon ma mes "
    "faut faire comment combien quel quels quelle quelles fait peut puis".split()
)


def norm(s):
    s = unicodedata.normalize("NFD", s.lower().replace("’", "'").replace("œ", "oe").replace("æ", "ae"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", norm(text)) if len(t) > 1 and t not in STOPWORDS]


def short_title(title):
    return title.split("/")[-1].strip()


def load_docs():
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CORPUS_DIR.glob("*.json"))]


def chunk_doc(doc):
    """Découpe par lignes, en regroupant jusqu'à ~CHUNK_CHARS caractères. Les recettes courtes
    restent en un seul extrait ; le titre est répété dans chaque extrait pour aider la recherche."""
    lines = [l.strip() for l in doc["text"].splitlines() if l.strip()]
    chunks, cur = [], ""
    for line in lines:
        if cur and len(cur) + len(line) + 1 > CHUNK_CHARS:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return [{"doc_id": doc["id"], "title": doc["title"], "url": doc.get("url", ""),
             "text": f"{short_title(doc['title'])}\n{c}"} for c in chunks]


class Index:
    def __init__(self):
        docs = load_docs()
        if not docs:
            raise RuntimeError(f"Aucun document dans {CORPUS_DIR} : lancez scripts/fetch_wikibooks.py")
        self.chunks = [c for d in docs for c in chunk_doc(d)]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.vecs = None
        if MODE in ("hybrid", "dense"):
            self.vecs = self._embeddings()

    def _embeddings(self):
        key = hashlib.sha1(
            (llm.EMBED_MODEL + str(CHUNK_CHARS) + "|".join(c["text"] for c in self.chunks)).encode("utf-8")
        ).hexdigest()[:16]
        path = INDEX_DIR / f"emb_{key}.npy"
        if path.exists():
            return np.load(path)
        print(f"[niveau 1] calcul des embeddings ({len(self.chunks)} extraits, modèle {llm.EMBED_MODEL})...")
        v = np.array(llm.embed([c["text"] for c in self.chunks]), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, v)
        return v


_INDEX = None


def get_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = Index()
    return _INDEX


def warmup():
    """Construit l'index avant la première question, pour que la latence mesurée ne l'inclue pas."""
    get_index()


def retrieve(query, n=N_CANDIDATES):
    """Retourne (liste d'indices d'extraits classés, meilleure similarité dense ou None)."""
    idx = get_index()
    fused, best_sim = {}, None

    def add(order):
        for rank, i in enumerate(order):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)

    if MODE in ("hybrid", "bm25"):
        scores = idx.bm25.get_scores(tokenize(query))
        if scores.max() > 0:
            add([i for i in np.argsort(-scores)[:n] if scores[i] > 0])
    if MODE in ("hybrid", "dense"):
        q = np.array(llm.embed([query])[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        sims = idx.vecs @ q
        best_sim = float(sims.max())
        add(np.argsort(-sims)[:n])
    return sorted(fused, key=lambda i: -fused[i])[:n], best_sim


def doc_ranking(chunk_ids):
    """Classement au niveau des documents (premier passage de chaque document)."""
    idx = get_index()
    seen, out = set(), []
    for i in chunk_ids:
        d = idx.chunks[i]["doc_id"]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def build_context(chunk_ids):
    idx = get_index()
    return "\n\n".join(
        f"[{n}] {short_title(idx.chunks[i]['title'])} (id : {idx.chunks[i]['doc_id']})\n{idx.chunks[i]['text']}"
        for n, i in enumerate(chunk_ids, 1)
    )


def cited_docs(answer, chunk_ids):
    idx = get_index()
    out = []
    for m in re.findall(r"\[(\d+)\]", answer):
        n = int(m)
        if 1 <= n <= len(chunk_ids):
            d = idx.chunks[chunk_ids[n - 1]]["doc_id"]
            if d not in out:
                out.append(d)
    return out


def answer(question: str) -> dict:
    t0 = time.time()
    ranked, best_sim = retrieve(question)
    retrieved = doc_ranking(ranked)
    trace = {"mode": MODE, "retrieved_docs": retrieved[:N_CANDIDATES], "best_sim": best_sim}

    if MIN_SIM > 0 and best_sim is not None and best_sim < MIN_SIM:
        return {"answer": ABSTAIN, "sources": [], "retrieved": retrieved, "state": {}, "steps": [trace],
                "tokens_in": 0, "tokens_out": 0, "latency": time.time() - t0}

    context_ids = ranked[:TOP_K]
    r = llm.chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Extraits :\n\n{build_context(context_ids)}\n\nQuestion : {question}"},
    ])
    text = r["message"].get("content") or ""
    return {
        "answer": text,
        "sources": cited_docs(text, context_ids),
        "retrieved": retrieved,
        "state": {},
        "steps": [trace],
        "tokens_in": r["tokens_in"],
        "tokens_out": r["tokens_out"],
        "latency": time.time() - t0,
    }
