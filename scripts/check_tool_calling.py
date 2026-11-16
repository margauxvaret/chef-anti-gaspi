"""Le modèle sait-il appeler les outils de l'agent ? Mesure rapide avant d'écrire le niveau 2.

Passe par le même chemin que l'agent : src/llm.py (API OpenAI-compatible d'Ollama ou du prof) et les vrais
schémas de src/tools.py, plus les trois outils prévus pour le niveau 2 (recherche de recettes, ajout et envoi
de la liste). Rien n'est exécuté sauf convertir_unites, sans réseau, pour vérifier le résultat obtenu.

À lancer depuis la racine du dépôt :
    python scripts/check_tool_calling.py                                  # modèle de .env (LLM_MODEL)
    python scripts/check_tool_calling.py --models qwen2.5:3b qwen2.5:7b --trials 3
    python scripts/check_tool_calling.py --cases envoi_liste               # un seul cas

Pour chaque modèle : réussite par essai, pass^k par cas, appels natifs ou JSON écrit dans le texte,
outils appelés à tort, envois non demandés, tokens et latence. Brut : runs/tool_check_<modèle>.jsonl
(utile pour justifier le choix du modèle dans le rapport).
"""
import argparse
import json
import statistics
import sys
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import llm  # noqa: E402
from src.tools import SCHEMAS, executer  # noqa: E402


def schema(nom, description, proprietes, requis):
    return {"type": "function", "function": {"name": nom, "description": description, "parameters": {
        "type": "object", "properties": proprietes, "required": requis}}}


# Outils prévus pour le niveau 2 (à reprendre dans src/level2.py).
OUTILS_NIVEAU2 = [
    schema("rechercher_recettes",
           "Cherche dans le livre de cuisine (Wikilivres) les extraits de recettes utiles : ingrédients, quantités, étapes.",
           {"requete": {"type": "string", "description": "Plat ou ingrédients recherchés, ex. pipérade"}}, ["requete"]),
    schema("ajouter_a_la_liste", "Ajoute des articles à la liste de courses (action modifiable).",
           {"articles": {"type": "array", "items": {"type": "object", "properties": {
               "nom": {"type": "string"}, "quantite": {"type": "number"}, "unite": {"type": "string"}},
               "required": ["nom"]}}}, ["articles"]),
    schema("envoyer_liste",
           "Envoie la liste de courses. Action irréversible : elle ne part qu'après confirmation de l'utilisateur.",
           {}, []),
]
TOOLS = SCHEMAS + OUTILS_NIVEAU2

SYSTEM = (
    "Tu es un chef anti-gaspi. Tu aides l'utilisateur à cuisiner avec ses restes et à préparer sa liste de courses. "
    "Quand une demande nécessite un outil, appelle-le. Sinon, réponds directement en français, sans outil. "
    "Si la demande sort du domaine de la cuisine, refuse poliment."
)


def norm(s):
    s = unicodedata.normalize("NFD", str(s or "").lower().replace("œ", "oe"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def nombre(x):
    try:
        return float(str(x).replace(",", "."))
    except (TypeError, ValueError):
        return None


def conversion(attendu):
    """Exécute vraiment convertir_unites avec les arguments du modèle et compare le résultat."""
    def check(appels):
        for nom, a in appels:
            if nom == "convertir_unites":
                r = executer(nom, a)
                if r.get("ok") and abs(r["resultat"] - attendu) < 1e-6:
                    return True
        return False
    return "convertir_unites", check


def contient(outil, champ, mot):
    return outil, lambda appels: any(n == outil and mot in norm(a.get(champ)) for n, a in appels)


def articles(attendus):
    def check(appels):
        tous = []
        for n, a in appels:
            if n == "ajouter_a_la_liste":
                liste = a.get("articles")
                if isinstance(liste, str):  # tableau renvoyé en texte par certains petits modèles
                    try:
                        liste = json.loads(liste)
                    except json.JSONDecodeError:
                        liste = []
                tous += [x for x in (liste or []) if isinstance(x, dict)]
        return all(any(mot in norm(x.get("nom")) and nombre(x.get("quantite")) == q for x in tous)
                   for mot, q in attendus.items())
    return "ajouter_a_la_liste", check


AUCUN = (None, lambda appels: not appels)

CASES = [
    ("conv_cl_ml", "Ma recette demande 20 cl de vin blanc, ça fait combien en millilitres ?", conversion(200)),
    ("conv_cas_ml", "3 cuillères à soupe d'huile, ça fait combien de millilitres ?", conversion(45)),
    ("conv_thermostat", "Le four doit être à 220 °C, ça correspond à quel thermostat ?", conversion(7)),
    ("produit_coco", "Cherche un lait de coco en boîte dans Open Food Facts.", contient("chercher_produit", "nom", "coco")),
    ("recette_restes", "Que puis-je cuisiner avec deux courgettes et un reste de riz ?",
     contient("rechercher_recettes", "requete", "courgette")),
    ("recette_nommee", "Quels ingrédients faut-il pour la pipérade ?", contient("rechercher_recettes", "requete", "piperade")),
    ("ajout_2_articles", "Ajoute 6 œufs et 500 g de farine à ma liste de courses.", articles({"oeuf": 6, "farine": 500})),
    ("envoi_liste", "Ma liste de courses contient 6 œufs et 500 g de farine. C'est bon, envoie-la.",
     ("envoyer_liste", lambda appels: any(n == "envoyer_liste" for n, _ in appels))),
    ("merci", "Merci beaucoup, c'était très utile !", AUCUN),
    ("hors_sujet", "Quel est le cours du Bitcoin aujourd'hui ?", AUCUN),
]


def lire_appels(message):
    appels = []
    for tc in message.get("tool_calls") or []:
        f = tc.get("function") or {}
        args = f.get("arguments") or {}
        if isinstance(args, str):  # format OpenAI : arguments = texte JSON
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        appels.append((f.get("name"), args if isinstance(args, dict) else {}))
    return appels


def format_sortie(message, appels):
    if appels:
        return "natif"
    texte = (message.get("content") or "").strip()
    if "<tool_call>" in texte or texte.startswith(("{", "```")) or '"name"' in texte:
        return "json_dans_texte"  # l'appel est écrit dans la réponse au lieu d'être fait
    return "texte"


def evaluer(modele, essais, temperature, cas_choisis=None):
    lignes = []
    for cas_id, question, (attendu, check) in CASES:
        if cas_choisis and cas_id not in cas_choisis:
            continue
        for essai in range(1, essais + 1):
            r = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
                         tools=TOOLS, temperature=temperature, model=modele, seed=essai)
            message = r["message"] or {}
            appels = lire_appels(message)
            envoi_non_demande = attendu != "envoyer_liste" and any(n == "envoyer_liste" for n, _ in appels)
            ok = check(appels) and not envoi_non_demande
            ligne = {"modele": modele, "cas": cas_id, "essai": essai, "attendu": attendu, "ok": ok,
                     "format": format_sortie(message, appels),
                     "appels": [{"outil": n, "arguments": a} for n, a in appels],
                     "texte": (message.get("content") or "")[:300], "envoi_non_demande": envoi_non_demande,
                     "tokens": r["tokens_in"] + r["tokens_out"], "latence_s": round(r["latency"], 2)}
            lignes.append(ligne)
            print(f"  {cas_id:<17} essai {essai}  {'OK' if ok else 'ÉCHEC':<6}{ligne['format']:<16}"
                  f"{', '.join(n for n, _ in appels) or '-'}")
    return lignes


def resumer(modele, lignes):
    par_cas = {}
    for l in lignes:
        par_cas.setdefault(l["cas"], []).append(l["ok"])
    avec = [l for l in lignes if l["attendu"]]
    sans = [l for l in lignes if not l["attendu"]]
    return {"modele": modele, "reussite": sum(l["ok"] for l in lignes) / len(lignes),
            "pass_k": sum(all(v) for v in par_cas.values()), "nb_cas": len(par_cas),
            "natif": sum(l["format"] == "natif" for l in avec) / max(len(avec), 1),
            "json_texte": sum(l["format"] == "json_dans_texte" for l in lignes),
            "inutiles": sum(bool(l["appels"]) for l in sans) / max(len(sans), 1),
            "envois": sum(l["envoi_non_demande"] for l in lignes),
            "tokens": statistics.mean(l["tokens"] for l in lignes),
            "latence": statistics.median(l["latence_s"] for l in lignes),
            "echecs": [c for c, v in par_cas.items() if not all(v)]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=[llm.MODEL])
    p.add_argument("--trials", type=int, default=3, help="essais par cas (pass^k)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="0.7 par défaut pour que les essais diffèrent ; 0 = comportement déterministe")
    p.add_argument("--cases", nargs="+", choices=[c[0] for c in CASES], help="ne lancer que ces cas")
    args = p.parse_args()
    nb_cas = len(args.cases) if args.cases else len(CASES)

    resumes = []
    for modele in args.models:
        print(f"\n=== {modele} : {nb_cas} cas x {args.trials} essais, température {args.temperature}")
        try:
            lignes = evaluer(modele, args.trials, args.temperature, args.cases)
        except requests.HTTPError as e:
            detail = e.response.text if e.response is not None else str(e)
            print(f"  Erreur du serveur : {detail[:300]}")
            if "does not support tools" in detail:
                print("  -> ce modèle ne gère pas les appels d'outils.")
            elif e.response is not None and e.response.status_code == 404:
                print(f"  -> modèle absent : ollama pull {modele}")
            continue
        except requests.ConnectionError:
            print(f"  Pas de serveur sur {llm.BASE_URL} : lancez Ollama (`ollama serve`) ou vérifiez .env.")
            return 1
        sortie = ROOT / "runs" / f"tool_check_{modele.replace(':', '_').replace('/', '_')}.jsonl"
        sortie.parent.mkdir(exist_ok=True)
        with open(sortie, "w", encoding="utf-8") as f:
            for l in lignes:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
        resumes.append(resumer(modele, lignes))
        print(f"  -> {sortie.relative_to(ROOT)}")

    if not resumes:
        return 1
    k = args.trials
    print(f"\n{'modèle':<16}{'réussite':>9}{f'pass^{k}':>9}{'natif':>8}{'JSON texte':>11}"
          f"{'à tort':>8}{'envois !':>10}{'tokens':>8}{'latence':>9}")
    for r in resumes:
        print(f"{r['modele']:<16}{r['reussite']:>9.0%}{r['pass_k']:>6}/{r['nb_cas']:<2}{r['natif']:>8.0%}"
              f"{r['json_texte']:>11}{r['inutiles']:>8.0%}{r['envois']:>10}{r['tokens']:>8.0f}{r['latence']:>8.1f}s")
    for r in resumes:
        if r["echecs"]:
            print(f"  {r['modele']} échoue sur : {', '.join(r['echecs'])}")
    print("\nnatif = vrais appels d'outils ; JSON texte = appel écrit dans la réponse au lieu d'être fait ; "
          "à tort = outil appelé alors qu'il ne fallait pas ; envois ! = envoyer_liste appelé sans demande.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
