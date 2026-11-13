"""Évaluation commune à tous les niveaux : UN protocole, UN script.

    python eval.py                       # tous les niveaux dont src/level<N>.py existe
    python eval.py --levels 0 1          # certains niveaux seulement
    python eval.py --levels 2 --trials 3 # pass^3 sur les tâches multi-étapes

Sorties : runs/<horodatage>_niveau<N>.jsonl (résultats bruts + traces) et runs/results.md.

Chaque niveau expose answer(question) -> dict (voir src/level0.py).

ATTENTION : les scores « abstention » et « clarification » ci-dessous sont des heuristiques
sur des mots-clés. Elles servent à démarrer ; avant le rapport, remplacez-les par un juge LLM
calibré sur ~20 réponses annotées à la main (voir le sujet, section 2.3).
"""
import argparse
import importlib
import json
import statistics
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # permet « import src.levelN » quel que soit le dossier de lancement
TESTS = ROOT / "tests.jsonl"
RUNS = ROOT / "runs"

ABSTAIN_MARKERS = [
    "je ne sais pas", "je ne trouve pas", "pas dans les documents", "aucune information",
    "je n'ai pas d'information", "impossible de repondre", "je ne dispose pas",
    "je ne peux pas repondre", "n'est pas couvert",
]
CLARIFY_MARKERS = [
    "pouvez-vous preciser", "peux-tu preciser", "pourriez-vous preciser", "quel type",
    "combien de personnes", "pouvez-vous me dire", "peux-tu me dire", "plus de details",
    "hors sujet", "je ne peux pas vous aider", "specialise",
]


def norm(s: str) -> str:
    """Minuscules, sans accents, apostrophes droites."""
    s = unicodedata.normalize("NFD", s.lower().replace("’", "'"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def has_marker(text: str, markers) -> bool:
    t = norm(text)
    return any(m in t for m in markers)


def load_tests():
    cases = []
    with open(TESTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(json.loads(line))
    return cases


def check_state(spec: dict, state: dict) -> bool:
    """Vérifie l'état final par le CODE, jamais par le texte de la réponse."""
    items = [norm(str(x)) for x in state.get("shopping_list", [])]
    for wanted in spec.get("shopping_list_contains", []):
        if not any(norm(wanted) in it for it in items):
            return False
    for banned in spec.get("shopping_list_excludes", []):
        if any(norm(banned) in it for it in items):
            return False
    if "sent" in spec and bool(state.get("sent")) != spec["sent"]:
        return False
    return True


def score(case: dict, out: dict) -> dict:
    ans = out.get("answer", "")
    t = case["type"]
    s = {}
    if t == "in_docs":
        kws = case.get("expected_keywords", [])
        s["correct"] = bool(kws) and all(norm(k) in norm(ans) for k in kws) and not has_marker(ans, ABSTAIN_MARKERS)
        src = case.get("expected_source")
        if src and src != "TODO":
            retrieved = out.get("retrieved")
            if retrieved is not None:
                s["recall5"] = 1.0 if src in retrieved[:5] else 0.0
                s["rr"] = 1.0 / (retrieved.index(src) + 1) if src in retrieved else 0.0
    elif t == "out_of_docs":
        s["abstained"] = has_marker(ans, ABSTAIN_MARKERS)
    elif t == "multi_step":
        s["success"] = check_state(case.get("final_state", {}), out.get("state", {}))
    elif t == "ambiguous":
        s["clarified"] = "?" in ans or has_marker(ans, CLARIFY_MARKERS)
    return s


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def fmt(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def run_level(level: int, cases, trials: int):
    mod = importlib.import_module(f"src.level{level}")
    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RUNS / f"{stamp}_niveau{level}.jsonl"
    rows = []
    with open(path, "w", encoding="utf-8") as f:
        for case in cases:
            n = trials if case["type"] == "multi_step" else 1
            for trial in range(n):
                try:
                    out = mod.answer(case["question"])
                    err = None
                except Exception as e:  # une panne compte comme un échec, pas comme un crash
                    out, err = {"answer": "", "state": {}}, repr(e)
                sc = score(case, out)
                row = {"level": level, "case_id": case["id"], "type": case["type"],
                       "trial": trial, "question": case["question"], "error": err,
                       "score": sc, **out}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  niveau {level} | {case['id']} | essai {trial + 1}/{n} | {sc}")
    print(f"  -> {path}")
    return rows


def aggregate(rows, trials):
    def col(type_, key):
        return [r["score"].get(key) for r in rows if r["type"] == type_ and key in r["score"]]

    # pass^k : une tâche compte seulement si TOUS les essais réussissent
    by_task = {}
    for r in rows:
        if r["type"] == "multi_step":
            by_task.setdefault(r["case_id"], []).append(bool(r["score"].get("success")))
    first_try = [v[0] for v in by_task.values()]
    pass_k = [all(v) for v in by_task.values()] if trials > 1 else None

    return {
        "recall5": mean(col("in_docs", "recall5")),
        "mrr": mean(col("in_docs", "rr")),
        "exactitude": mean(col("in_docs", "correct")),
        "abstention": mean(col("out_of_docs", "abstained")),
        "clarif": mean(col("ambiguous", "clarified")),
        "succes": mean(first_try) if first_try else None,
        "pass_k": mean(pass_k) if pass_k else None,
        "tokens": mean([r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in rows]),
        "latence": mean([r.get("latency", 0) for r in rows]),
        "erreurs": sum(1 for r in rows if r["error"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", type=int)
    ap.add_argument("--trials", type=int, default=1, help="essais par tâche multi-étapes (3 pour pass^3)")
    args = ap.parse_args()

    levels = args.levels
    if levels is None:
        levels = sorted(int(p.stem.removeprefix("level")) for p in (ROOT / "src").glob("level[0-9].py"))

    cases = load_tests()
    counts = {t: sum(1 for c in cases if c["type"] == t) for t in {c["type"] for c in cases}}
    print(f"{len(cases)} cas de test : {counts}")

    results = {}
    for lv in levels:
        print(f"== Niveau {lv}")
        results[lv] = aggregate(run_level(lv, cases, args.trials), args.trials)

    k = f"pass^{args.trials}" if args.trials > 1 else "pass^k"
    lines = [
        f"| Niveau | Recall@5 | MRR | Exactitude | Abstention | Clarif. | Succès tâches | {k} | Tokens/requête | Latence (s) | Erreurs |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lv, a in results.items():
        lines.append(
            f"| {lv} | {fmt(a['recall5'])} | {fmt(a['mrr'])} | {fmt(a['exactitude'])} | {fmt(a['abstention'])} "
            f"| {fmt(a['clarif'])} | {fmt(a['succes'])} | {fmt(a['pass_k'])} | {fmt(a['tokens'], 0)} "
            f"| {fmt(a['latence'], 1)} | {a['erreurs']} |"
        )
    table = "\n".join(lines)
    (RUNS / "results.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
