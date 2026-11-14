"""Télécharge des pages Wikilivres (fr.wikibooks.org) en texte brut -> data/corpus/<id>.json

Exemples :
    python scripts/fetch_wikibooks.py --category "Catégorie:Recettes de cuisine" --depth 1 --limit 150
    python scripts/fetch_wikibooks.py --category "Catégorie:Cuisine italienne" "Catégorie:Cuisine provençale"
    python scripts/fetch_wikibooks.py --prefix "Livre de cuisine/" --limit 150

Le script est REPRENABLE : les pages déjà téléchargées sont ignorées, donc en cas d'arrêt
(limite de l'API, coupure réseau) il suffit de relancer la même commande.
Il ralentit tout seul quand Wikilivres répond « trop de requêtes » (erreur 429).

--depth N : descend aussi dans les sous-catégories sur N niveaux (0 = la catégorie seule).

Licence : le contenu de Wikilivres est sous CC BY-SA. Chaque document garde son titre et son URL,
à citer dans le rapport. Le dossier data/corpus est à COMMITER pour que le dépôt soit reproductible.
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

API = "https://fr.wikibooks.org/w/api.php"
HEADERS = {"User-Agent": "chef-anti-gaspi-projet-etudiant/0.1 (cours IA generative)"}
OUT = Path(__file__).resolve().parent.parent / "data" / "corpus"
PAUSE = 1.0  # secondes entre deux pages


def api(params, retries=8):
    for attempt in range(retries):
        r = requests.get(API, params={"format": "json", "formatversion": 2, **params},
                         headers=HEADERS, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            try:
                wait = int(r.headers.get("Retry-After", ""))
            except ValueError:
                wait = 0
            wait = max(wait, 5 * (attempt + 1))
            print(f"  limite atteinte (HTTP {r.status_code}), pause de {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("API Wikilivres indisponible, relancez la commande dans quelques minutes.")


def category_pages(category, depth, seen):
    pages, subcats, cont = [], [], {}
    while True:
        data = api({"action": "query", "list": "categorymembers", "cmtitle": category,
                    "cmtype": "page|subcat", "cmlimit": "max", **cont})
        for m in data["query"]["categorymembers"]:
            (subcats if m["ns"] == 14 else pages).append(m["title"])
        if "continue" not in data:
            break
        cont = data["continue"]
    if depth > 0:
        for sc in subcats:
            if sc not in seen:
                seen.add(sc)
                pages += category_pages(sc, depth - 1, seen)
    return pages


def prefix_pages(prefix):
    pages, cont = [], {}
    while True:
        data = api({"action": "query", "list": "allpages", "apprefix": prefix,
                    "aplimit": "max", **cont})
        pages += [m["title"] for m in data["query"]["allpages"]]
        if "continue" not in data:
            break
        cont = data["continue"]
    return pages


def fetch(title):
    data = api({"action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "plain",
                "inprop": "url", "redirects": 1, "titles": title})
    page = data["query"]["pages"][0]
    return page.get("extract") or "", page.get("fullurl", "")


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", nargs="+", help="une ou plusieurs catégories")
    ap.add_argument("--prefix")
    ap.add_argument("--depth", type=int, default=0, help="niveaux de sous-catégories à parcourir")
    ap.add_argument("--limit", type=int, default=150, help="nombre maximal de documents dans data/corpus")
    ap.add_argument("--min-chars", type=int, default=300, help="ignore les pages trop courtes")
    args = ap.parse_args()
    if not (args.category or args.prefix):
        ap.error("donnez --category ou --prefix")

    OUT.mkdir(parents=True, exist_ok=True)
    titles, seen = [], set()
    for cat in args.category or []:
        titles += category_pages(cat, args.depth, seen)
    if args.prefix:
        titles += prefix_pages(args.prefix)
    titles = list(dict.fromkeys(titles))  # sans doublons, ordre conservé
    print(f"{len(titles)} pages trouvées")

    have = {p.stem for p in OUT.glob("*.json")}
    kept = len(have)
    for i, t in enumerate(titles, 1):
        if kept >= args.limit:
            break
        doc_id = slug(t)
        if doc_id in have:
            continue
        text, url = fetch(t)
        time.sleep(PAUSE)  # politesse envers l'API
        if len(text) < args.min_chars:
            print(f"  [{i}/{len(titles)}] ignorée (trop courte) : {t}")
            continue
        doc = {"id": doc_id, "title": t, "url": url, "license": "CC BY-SA (Wikilivres)", "text": text}
        (OUT / f"{doc_id}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        have.add(doc_id)
        kept += 1
        print(f"  [{i}/{len(titles)}] ok ({kept}/{args.limit}) : {t}")
    print(f"{kept} documents dans {OUT}")


if __name__ == "__main__":
    main()