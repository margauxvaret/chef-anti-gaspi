"""Résume un run d'eval.py cas par cas : succès, raison de l'arrêt, outils appelés, contexte, liste finale.

    python scripts/resume_run.py                     # dernier run du niveau 2
    python scripts/resume_run.py runs/xxx.jsonl      # un run précis
    python scripts/resume_run.py --cas multi-02      # le détail d'un seul cas (arguments et réponse complète)

Sert à l'analyse des échecs : on voit à quelle étape l'agent s'est arrêté et pourquoi.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def court(x, n=90):
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fichier", nargs="?")
    p.add_argument("--cas", help="id d'un cas (ex. multi-02) pour voir tout le détail")
    args = p.parse_args()

    fichier = Path(args.fichier) if args.fichier else max((ROOT / "runs").glob("*_niveau2.jsonl"), default=None)
    if fichier is None or not fichier.exists():
        print("Aucun run du niveau 2 dans runs/ : lancez d'abord python eval.py --levels 2")
        return 1
    lignes = [json.loads(l) for l in open(fichier, encoding="utf-8") if l.strip()]
    if args.cas:
        lignes = [r for r in lignes if r["case_id"] == args.cas]
    print(f"{fichier.name} : {len(lignes)} lignes\n")

    for r in lignes:
        succes = r["score"].get("success", r["score"].get("correct"))
        st = r.get("state") or {}
        print(f"== {r['case_id']} essai {r['trial'] + 1}  succès={succes}  arrêt={r.get('arret')}  "
              f"contexte_max={r.get('contexte_max')}  tokens={r.get('tokens_in', 0) + r.get('tokens_out', 0)}  "
              f"latence={r.get('latency', 0):.0f}s")
        if r.get("error"):
            print(f"   ERREUR : {r['error']}")
        print(f"   question : {court(r['question'], 120)}")
        for s in r.get("steps", []):
            if "outils" not in s:
                print(f"   étape {s.get('etape')} : {s.get('type')} {court(s.get('decision', ''))}")
                continue
            outils = s["outils"]
            if not outils:
                print(f"   étape {s['etape']} (in {s.get('tokens_in')}) : réponse texte {s.get('remarque', '')}")
            for o in outils:
                res = o.get("resultat") or {}
                etat = "ok" if res.get("ok") else f"ERREUR {court(res.get('erreur', ''), 60)}"
                detail = o["arguments"] if args.cas else court(o["arguments"], 70)
                print(f"   étape {s['etape']} (in {s.get('tokens_in')}) : {o['outil']} [{o.get('effet')}] {detail} -> {etat}")
        print(f"   liste : {st.get('shopping_list')}  envoyée={st.get('sent')}  en attente={st.get('pending_confirmation')}")
        print(f"   réponse : {r.get('answer', '') if args.cas else court(r.get('answer', ''), 150)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
