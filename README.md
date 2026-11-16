# Chef anti-gaspi

Assistant de cuisine anti-gaspi (parcours Texte, projet « Mineur IA générative » 2026, groupe 03).
Il répond à partir d'un corpus de recettes Wikilivres, propose des idées avec les restes du frigo et prépare
une liste de courses. Quatre niveaux sont comparés sur **le même protocole de test** :

| Niveau | Description | Fichier | État |
|---|---|---|---|
| 0 | LLM seul (référence) | `src/level0.py` | fait |
| 1 | RAG sur des recettes Wikilivres (BM25 + embeddings, citations, abstention) | `src/level1.py` | fait |
| 2 | Agent outillé (Open Food Facts, conversion d'unités, liste de courses) | `src/level2.py` | outils prêts (`src/tools.py`), agent en cours |
| 3 | Extension mesurée contre le niveau 2 | `src/level3.py` | à faire |

Action à valider par l'utilisateur : l'envoi de la liste de courses (rien n'est envoyé sans confirmation explicite).

## Installation

Windows (PowerShell) :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # puis adapter
```

Linux / macOS :

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # puis adapter
```

Aucune clé d'API n'est dans le dépôt : `.env` est ignoré par git, seul `.env.example` est versionné.

### Modèle local avec Ollama (développement)

```bash
ollama pull qwen2.5:3b        # ou un autre petit modèle ; pour le niveau 2, choisir un modèle
                              # qui gère les appels d'outils (filtre « tools » sur ollama.com/search)
ollama pull bge-m3            # modèle d'embeddings multilingue (niveau 1)
ollama serve                  # si pas déjà lancé
```

Le code parle le protocole OpenAI-compatible (`src/llm.py`). Pour utiliser l'API partagée du prof, changer
`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` dans `.env`, rien d'autre.

> **Règle d'équité :** les résultats finaux des niveaux 0 à 3 doivent venir du **même modèle**.
> Ne comparez pas un niveau local à un niveau API.

Test sans modèle ni réseau (vérifie seulement que le code tourne) : `LLM_MODEL=mock` et `EMBED_MODEL=mock`
(PowerShell : `$env:LLM_MODEL="mock"; $env:EMBED_MODEL="mock"`).

## Données

- **Recettes** : 120 documents de Wikilivres (« Livre de cuisine »), dans `data/corpus/`, licence CC BY-SA
  (titre, URL et licence conservés dans chaque document).
  Reconstruction : `python scripts/fetch_wikibooks.py --category "Catégorie:..." --limit 150`
  (voir l'en-tête du script ; le script espace ses requêtes pour respecter les limites de Wikilivres).
- **Produits** : Open Food Facts (données ODbL), interrogé en direct par `src/tools.py`.

## Reproduire le tableau de résultats

```bash
python eval.py                          # tous les niveaux disponibles
python eval.py --levels 1               # un seul niveau
python eval.py --levels 2 --trials 3    # pass^3 sur les tâches multi-étapes
python eval.py --only multi_step        # un seul type de cas
```

Résultats bruts et traces : `runs/*.jsonl`. Tableau : `runs/results.md`.
Le premier lancement du niveau 1 calcule les embeddings (cache dans `data/index/`, non versionné).
Variantes du RAG pour le rapport : `RAG_MODE=bm25`, `RAG_MODE=dense` ou `RAG_MODE=hybrid`
(voir l'en-tête de `src/level1.py` pour tous les réglages).

Pour une évaluation sans réseau côté Open Food Facts : `OFF_OFFLINE=1` (n'utilise que `data/off_cache.json`,
versionné pour que le tableau soit reproductible).

### Résultats actuels (modèle local `qwen2.5:3b`, niveaux 0 et 1)

| Niveau | Recall@5 | MRR | Exactitude | Source citée | Abstention | Tokens/requête | Latence (s) |
|---|---|---|---|---|---|---|---|
| 0 LLM seul | n/a | n/a | 0.10 | n/a | 0.00 | 223 | 41.8 |
| 1 RAG hybride | 1.00 | 0.97 | 0.80 | 0.84 | 1.00 | 1019 | 55.4 |

Le RAG fait passer l'exactitude de 0.10 à 0.80 et l'abstention de 0.00 à 1.00, pour environ 4,5 fois plus de
tokens. Le tableau complet et à jour est dans `runs/results.md` ; les niveaux 2 et 3 y seront ajoutés.

## Outils de l'agent (`src/tools.py`)

- `convertir_unites(valeur, de, vers)` : masses, volumes, cuillères, températures et thermostat.
- `chercher_produit(nom)` : recherche Open Food Facts (Search-a-licious, avec repli sur l'API v1),
  espacée pour respecter la limite de 10 recherches par minute, réponses mises en cache.

Chaque outil renvoie toujours un dictionnaire (`{"ok": True, ...}` ou `{"ok": False, "erreur": "..."}`) et ne lève
jamais d'exception, pour que l'agent puisse renvoyer l'erreur au modèle. Branchement dans un agent :

```python
from src.tools import SCHEMAS, executer
```

Tests des outils (sans réseau) : `python -m unittest test_tools -v`.

## Protocole de test

`tests.jsonl` : 40 cas (20 réponse dans les documents, 5 hors documents, 10 multi-étapes, 5 ambigus).
Le format est documenté en tête du fichier. **On ne modifie pas le protocole sans prévenir l'autre membre du groupe.**

Métriques : Recall@5 et MRR (recherche), exactitude et source citée (réponses), abstention et demande de
clarification, succès des tâches multi-étapes (état final de la liste de courses), pass^k, tokens et latence.

## Organisation du dépôt

```
eval.py            protocole d'évaluation commun
tests.jsonl        jeu de test
test_tools.py      tests unitaires des outils
src/llm.py         client LLM unique
src/level*.py      un niveau = un fichier exposant answer(question) -> dict
src/tools.py       outils de l'agent (conversion d'unités, Open Food Facts)
scripts/           collecte des données
data/corpus/       documents du RAG
data/off_cache.json  cache Open Food Facts (évaluation reproductible)
runs/              résultats bruts + results.md
NOTES.md           échecs observés (matière de l'analyse de 5 échecs du rapport)
```

## Règles du groupe

- Une branche par personne, merge souvent.
- Jamais de clé d'API dans le dépôt (`.env` est ignoré).
- Chaque échec intéressant observé va dans `NOTES.md`.