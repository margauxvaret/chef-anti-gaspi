# Chef anti-gaspi

Assistant de cuisine anti-gaspi (parcours Texte). Quatre niveaux comparés sur le même protocole de test :

| Niveau | Description | Fichier |
|---|---|---|
| 0 | LLM seul (référence) | `src/level0.py` |
| 1 | RAG sur des recettes Wikilivres (BM25 + embeddings, citations, abstention) | `src/level1.py` |
| 2 | Agent outillé (Open Food Facts, conversion d'unités, liste de courses) | `src/level2.py` (à faire) |
| 3 | Extension mesurée contre le niveau 2 | `src/level3.py` (à faire) |

Action à valider par l'utilisateur : envoi de la liste de courses.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # puis adapter
```

### Modèle local avec Ollama (développement)

```bash
ollama pull qwen2.5:3b        # ou un autre petit modèle ; pour le niveau 2, choisir un modèle
                              # qui gère les appels d'outils (filtre « tools » sur ollama.com/search)
ollama pull bge-m3            # modèle d'embeddings multilingue (niveau 1)
ollama serve                  # si pas déjà lancé
```

Niveau 1 : `python eval.py --levels 1`. Variantes pour le rapport : `RAG_MODE=bm25`, `RAG_MODE=dense`
ou `RAG_MODE=hybrid` (voir l'en-tête de `src/level1.py` pour tous les réglages). Le premier lancement
calcule les embeddings (cache dans `data/index/`).

Le code parle le protocole OpenAI-compatible (`src/llm.py`). Pour utiliser l'API partagée du prof, changer
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` dans `.env`, rien d'autre.

> **Règle d'équité :** les résultats finaux des niveaux 0 à 3 doivent venir du **même modèle**.
> Ne comparez pas un niveau local à un niveau API.

## Données

```bash
python scripts/fetch_wikibooks.py --category "Catégorie:..." --limit 150   # voir l'en-tête du script
```

Le dossier `data/corpus/` est commité (licence CC BY-SA, titre et URL conservés dans chaque document).

## Reproduire le tableau de résultats

```bash
python eval.py                   # niveaux disponibles
python eval.py --levels 2 --trials 3   # pass^3 sur les tâches multi-étapes
```

Résultats bruts et traces : `runs/*.jsonl`. Tableau : `runs/results.md`.

## Protocole de test

`tests.jsonl` : 40 cas (20 réponse dans les documents, 5 hors documents, 10 multi-étapes, 5 ambigus).
Le format est documenté en tête du fichier. **On ne modifie pas le protocole sans prévenir l'autre membre du groupe.**

## Organisation du dépôt

```
eval.py            protocole d'évaluation commun
tests.jsonl        jeu de test
src/llm.py         client LLM unique
src/level*.py      un niveau = un fichier exposant answer(question) -> dict
scripts/           collecte des données
data/corpus/       documents du RAG
runs/              résultats bruts + results.md
```

## Règles du groupe

- Une branche par personne, merge souvent.
- Jamais de clé d'API dans le dépôt (`.env` est ignoré).
- Chaque échec intéressant observé va dans `NOTES.md` (de quoi remplir l'analyse de 5 échecs du rapport).
