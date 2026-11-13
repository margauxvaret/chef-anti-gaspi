"""Télécharge des pages Wikilivres (fr.wikibooks.org) en texte brut -> data/corpus/<slug>.json

Exemples :
    python scripts/fetch_wikibooks.py --category "Catégorie:NOM_EXACT" --limit 150
    python scripts/fetch_wikibooks.py --prefix "Livre de cuisine/" --limit 150

Trouvez d'abord le bon nom de catégorie ou de préfixe en parcourant le livre de cuisine sur
fr.wikibooks.org (je n'ai pas pu le vérifier depuis mon environnement, donc rien n'est codé en dur).

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


def api(params):
    r = requests.get(API, params={"format": "json", "formatversion": 2, **params}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def list_titles(category=None, prefix=None, limit=200):
    titles, cont = [], {}
    while len(titles) < limit:
        if category:
            data = api({"action": "query", "list": "categorymembers", "cmtitle": category,
                        "cmtype": "page", "cmlimit": "max", **cont})
            titles += [m["title"] for m in data["query"]["categorymembers"]]
        else:
            data = api({"action": "query", "list": "allpages", "apprefix": prefix,
                        "aplimit": "max", **cont})
            titles += [m["title"] for m in data["query"]["allpages"]]
        if "continue" not in data:
            break
        cont = data["continue"]
    return titles[:limit]


def fetch(title):
    data = api({"action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "plain",
                "inprop": "url", "redirects": 1, "titles": title})
    page = data["query"]["pages"][0]
    return page.get("extract") or "", page.get("fullurl", "")


def slug(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category")
    ap.add_argument("--prefix")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--min-chars", type=int, default=300, help="ignore les pages trop courtes")
    args = ap.parse_args()
    if not (args.category or args.prefix):
        ap.error("donnez --category ou --prefix")

    OUT.mkdir(parents=True, exist_ok=True)
    titles = list_titles(args.category, args.prefix, args.limit)
    print(f"{len(titles)} pages trouvées")
    kept = 0
    for t in titles:
        text, url = fetch(t)
        if len(text) < args.min_chars:
            continue
        doc = {"id": slug(t), "title": t, "url": url, "license": "CC BY-SA (Wikilivres)", "text": text}
        (OUT / f"{doc['id']}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        kept += 1
        time.sleep(0.2)  # politesse envers l'API
    print(f"{kept} documents écrits dans {OUT}")


if __name__ == "__main__":
    main()
