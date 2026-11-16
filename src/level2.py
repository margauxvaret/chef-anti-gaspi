"""Niveau 2 : agent outillé. La boucle est écrite à la main, sans framework.

Le modèle choisit les outils ; le CODE les exécute, applique les règles métier et garde la main sur
l'action irréversible (l'envoi de la liste), qui n'est jamais exécutée sans un « oui » de l'utilisateur.

Outils, classés par effet (dictionnaire EFFETS) :
    rechercher_recettes   lecture        extraits du livre de cuisine (RAG du niveau 1), cités [n]
    convertir_unites      lecture        src/tools.py
    chercher_produit      lecture        src/tools.py (Open Food Facts)
    ajouter_a_la_liste    écriture       modifiable, la liste vit dans l'état
    envoyer_liste         irréversible   jamais exécuté par le modèle : met l'envoi en attente de confirmation

Contrat commun : answer(question) -> dict (voir src/level0.py). En plus, answer(question, state) reprend
l'état renvoyé au tour précédent : c'est ce qui permet de répondre « oui » pour confirmer un envoi
(interface Streamlit). Sans état, chaque appel repart de zéro, ce dont eval.py a besoin.

Réglages (variables d'environnement) :
    LEVEL2_MAX_STEPS           appels au modèle par question (défaut 8)
    LEVEL2_MAX_TOKENS          tokens cumulés, entrée + sortie, par question (défaut 20000)
    LEVEL2_TEMPERATURE         défaut 0.7, avec une seed tirée au hasard (notée dans la sortie) : sans cela,
                               les 3 essais de pass^3 seraient identiques
    LEVEL2_TOP_K               extraits renvoyés par recherche (défaut 3 ; le contexte d'Ollama est limité)
    LEVEL2_RECHERCHE_INITIALE  1 = le code lance la recherche sur la question avant le premier appel au modèle
                               (variante à mesurer contre 0, le défaut, où le modèle décide seul)
"""
import copy
import json
import os
import random
import re
import time
import unicodedata
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from src import level1, llm
from src.tools import SCHEMAS as SCHEMAS_TOOLS
from src.tools import executer as executer_tool

ROOT = Path(__file__).resolve().parent.parent
ENVOIS = ROOT / "runs" / "envois.jsonl"  # l'envoi est simulé : chaque liste envoyée est ajoutée à ce journal

MAX_STEPS = int(os.getenv("LEVEL2_MAX_STEPS", "8"))
MAX_TOKENS = int(os.getenv("LEVEL2_MAX_TOKENS", "20000"))
TEMPERATURE = float(os.getenv("LEVEL2_TEMPERATURE", "0.7"))
TOP_K = int(os.getenv("LEVEL2_TOP_K", "3"))
RECHERCHE_INITIALE = os.getenv("LEVEL2_RECHERCHE_INITIALE", "0") == "1"
MAX_SORTIE = 1024     # tokens par réponse du modèle : une liste de 20 articles en JSON dépasse 512
MAX_ARTICLES = 60
HISTORIQUE = 6        # messages (questions et réponses) gardés d'un tour à l'autre

ABSTAIN = level1.ABSTAIN

EFFETS = {
    "rechercher_recettes": "lecture",
    "convertir_unites": "lecture",
    "chercher_produit": "lecture",
    "ajouter_a_la_liste": "ecriture",
    "envoyer_liste": "irreversible",
}


def _schema(nom, description, proprietes, requis):
    return {"type": "function", "function": {"name": nom, "description": description, "parameters": {
        "type": "object", "properties": proprietes, "required": requis}}}


SCHEMAS = [
    _schema("rechercher_recettes",
            "Cherche dans le livre de cuisine les extraits utiles : ingrédients, quantités, temps, températures. "
            "Chaque extrait a un numéro [n] à citer.",
            {"requete": {"type": "string", "description": "Plat ou ingrédients recherchés, ex. pipérade"}}, ["requete"]),
    *SCHEMAS_TOOLS,
    _schema("ajouter_a_la_liste",
            "Ajoute des articles à la liste de courses (action modifiable). Mettre tous les articles en un seul appel.",
            {"articles": {"type": "array", "items": {"type": "object", "properties": {
                "nom": {"type": "string", "description": "ex. oignons"},
                "quantite": {"type": "number", "description": "ex. 6 (facultatif)"},
                "unite": {"type": "string", "description": "ex. g, ml (facultatif)"}},
                "required": ["nom"]}}},
            ["articles"]),
    _schema("envoyer_liste",
            "Envoie la liste de courses. Action irréversible : le système demande lui-même la confirmation "
            "à l'utilisateur.", {}, []),
]
assert {s["function"]["name"] for s in SCHEMAS} == set(EFFETS), "chaque outil doit avoir un effet déclaré"

SYSTEM = f"""Tu es un chef anti-gaspi. Tu réponds en français, de façon concise, et tu utilises les outils.
Règles :
1. Question sur une recette (ingrédients, quantités, temps, température) : appelle rechercher_recettes, puis réponds UNIQUEMENT à partir des extraits, en citant chaque information par son numéro, ex. [1]. Si les extraits ne contiennent pas la réponse, réponds exactement : « {ABSTAIN} » N'invente jamais une recette absente des extraits.
2. Liste de courses : cherche la recette, adapte les quantités au nombre de personnes demandé, retire ce que l'utilisateur a déjà ou ne peut pas manger, puis appelle ajouter_a_la_liste avec tous les articles en un seul appel.
3. Envoi : si l'utilisateur demande d'envoyer la liste, appelle envoyer_liste. C'est le système qui demande la confirmation à l'utilisateur ; n'écris jamais que la liste est envoyée.
4. Conversion d'unités : appelle convertir_unites, ne calcule pas de tête.
5. Demande trop vague : pose une question de précision. Demande sans rapport avec la cuisine : refuse poliment.
Si un outil renvoie une erreur, corrige ta demande ou explique le problème à l'utilisateur."""


# ---------------------------------------------------------------------------------------------- utilitaires

def norm(s):
    s = unicodedata.normalize("NFD", str(s or "").lower().replace("’", "'").replace("œ", "oe").replace("æ", "ae"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _nombre(v):
    """Accepte 6, 1.5, '1,5', '1/2', '½'. Lève ValueError sinon."""
    if isinstance(v, bool):
        raise ValueError("booléen")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace(",", ".").replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4")
    if not s.split():
        raise ValueError("vide")
    try:
        return float(sum(Fraction(p) for p in s.split()))
    except ZeroDivisionError as e:
        raise ValueError("division par zéro") from e


def _fmt(x):
    return f"{x:g}".replace(".", ",")


def _cle(nom):
    """Clé de comparaison : « Oignons » et « oignon » désignent le même article."""
    return " ".join(m[:-1] if len(m) > 3 and m[-1] in "sx" else m for m in re.findall(r"[a-z0-9']+", norm(nom)))


def _libelle(article):
    if article.get("quantite") is None:
        return article["nom"]
    return f"{article['nom']} : {_fmt(article['quantite'])} {article.get('unite') or ''}".strip()


def _abreger(obj, n=200):
    """Version courte d'un résultat pour la trace (les extraits de recettes sont longs)."""
    if isinstance(obj, str):
        return obj if len(obj) <= n else obj[:n] + "…"
    if isinstance(obj, list):
        return [_abreger(x, n) for x in obj]
    if isinstance(obj, dict):
        return {k: _abreger(v, n) for k, v in obj.items()}
    return obj


class _Execution:
    """Tout ce qui concerne une seule question : état, extraits numérotés, trace, compteurs."""

    def __init__(self, state):
        self.state = state
        self.extraits = []        # ids d'extraits (niveau 1) dans l'ordre de leur numéro [n]
        self.retrieved = None     # classement des documents de la première recherche (Recall@5, MRR)
        self.steps = []
        self.tokens_in = self.tokens_out = self.contexte_max = 0
        self.deja_vus = set()     # appels déjà exécutés (outil + arguments)


def _etat(state):
    st = copy.deepcopy(state) if state else {}
    st.setdefault("articles", [])
    st.setdefault("sent", False)
    st.setdefault("pending_confirmation", False)
    st.setdefault("historique", [])
    st["shopping_list"] = [_libelle(a) for a in st["articles"]]  # format lu par eval.py : liste de textes
    return st


# ---------------------------------------------------------------------------------------------- outils

def _rechercher(args, ex):
    requete = str(args.get("requete") or "").strip()
    if len(requete) < 2:
        return {"ok": False, "erreur": "Requête vide : donner un plat ou des ingrédients, ex. « pipérade »."}
    ranked, best_sim = level1.retrieve(requete)
    if ex.retrieved is None:
        ex.retrieved = level1.doc_ranking(ranked)
    if not ranked or (level1.MIN_SIM > 0 and best_sim is not None and best_sim < level1.MIN_SIM):
        return {"ok": True, "extraits": [],
                "remarque": f"Aucun extrait pertinent. Pour une question de recette, réponds « {ABSTAIN} »"}
    chunks = level1.get_index().chunks
    extraits = []
    for i in ranked[:TOP_K]:
        if i not in ex.extraits:
            ex.extraits.append(i)
        extraits.append({"n": ex.extraits.index(i) + 1, "recette": level1.short_title(chunks[i]["title"]),
                         "texte": chunks[i]["text"]})
    return {"ok": True, "extraits": extraits,
            "consigne": "Cite chaque information par le numéro de son extrait, ex. [1]. "
                        "Si la réponse n'est pas dans les extraits, dis-le."}


def _ajouter(args, st):
    articles = args.get("articles")
    if isinstance(articles, str):  # certains petits modèles envoient le tableau sous forme de texte
        try:
            articles = json.loads(articles)
        except json.JSONDecodeError:
            pass
    if isinstance(articles, dict):
        articles = [articles]
    if not isinstance(articles, list) or not articles:
        return {"ok": False, "erreur": 'Donner une liste non vide, ex. [{"nom": "oignons", "quantite": 6}].'}

    remarques = []
    if st["sent"]:
        st["articles"], st["sent"] = [], False
        remarques.append("La liste précédente a déjà été envoyée : une nouvelle liste est commencée.")
    if st["pending_confirmation"]:
        st["pending_confirmation"] = False  # la liste change : la confirmation demandée ne vaut plus
        remarques.append("La liste a changé : l'envoi en attente est annulé, il faudra le redemander.")

    ajoutes, refuses = [], []
    for a in articles:
        if isinstance(a, str):
            a = {"nom": a}
        nom = str(a.get("nom") or "").strip() if isinstance(a, dict) else ""
        if not nom:
            refuses.append({"article": a, "raison": "nom manquant"})
            continue
        q = a.get("quantite")
        if q in (None, ""):
            q = None
        else:
            try:
                q = _nombre(q)
            except ValueError:
                refuses.append({"article": nom, "raison": f"quantité non numérique : {q!r}"})
                continue
            if q <= 0:
                refuses.append({"article": nom, "raison": "la quantité doit être positive"})
                continue
        unite = str(a.get("unite") or "").strip()
        # règle : même produit et même unité -> une seule ligne, quantités additionnées
        existant = next((x for x in st["articles"]
                         if _cle(x["nom"]) == _cle(nom) and norm(x.get("unite")) == norm(unite)), None)
        if existant:
            if q is not None:
                existant["quantite"] = (existant["quantite"] or 0) + q
            ajoutes.append(_libelle(existant) + " (regroupé)")
        elif len(st["articles"]) >= MAX_ARTICLES:
            refuses.append({"article": nom, "raison": f"liste pleine ({MAX_ARTICLES} articles maximum)"})
        else:
            st["articles"].append({"nom": nom, "quantite": q, "unite": unite})
            ajoutes.append(_libelle(st["articles"][-1]))

    st["shopping_list"] = [_libelle(a) for a in st["articles"]]
    res = {"ok": bool(ajoutes), "ajoutes": ajoutes, "liste": st["shopping_list"]}
    if refuses:
        res["refuses"] = refuses
    if not ajoutes:
        res["erreur"] = "Aucun article ajouté : voir « refuses »."
    if remarques:
        res["remarques"] = remarques
    return res


def _demander_confirmation(st):
    """Ce que reçoit le modèle quand il appelle envoyer_liste : rien n'est envoyé."""
    if not st["articles"]:
        return {"ok": False, "erreur": "La liste de courses est vide : il n'y a rien à envoyer."}
    if st["sent"]:
        return {"ok": False, "erreur": "Cette liste a déjà été envoyée."}
    st["pending_confirmation"] = True
    return {"ok": True, "statut": "en_attente_de_confirmation", "liste": st["shopping_list"],
            "message": "Rien n'est envoyé pour l'instant : l'utilisateur doit répondre « oui » pour confirmer. "
                       "Résume la liste et dis-lui de confirmer."}


def _envoyer(st):
    """L'action irréversible elle-même, appelée UNIQUEMENT par le code après un « oui »."""
    ENVOIS.parent.mkdir(exist_ok=True)
    with open(ENVOIS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"),
                            "liste": st["shopping_list"]}, ensure_ascii=False) + "\n")
    st["sent"], st["pending_confirmation"] = True, False
    return {"ok": True, "statut": "envoyee", "nombre_articles": len(st["articles"])}


def _executer(nom, args, ex):
    """Exécute un appel d'outil du modèle. Ne lève jamais d'exception : l'erreur repart au modèle."""
    cle = (nom, json.dumps(args, sort_keys=True, ensure_ascii=False))
    if cle in ex.deja_vus:
        return {"ok": False, "erreur": "Appel identique déjà fait : utilise le résultat précédent au lieu de le répéter."}
    ex.deja_vus.add(cle)
    effet = EFFETS.get(nom)
    if effet is None:
        return {"ok": False, "erreur": f"Outil inconnu : {nom!r}. Outils disponibles : {', '.join(EFFETS)}."}
    try:
        if effet == "irreversible":  # règle générale : une action irréversible attend toujours un « oui »
            return _demander_confirmation(ex.state)
        if nom == "rechercher_recettes":
            return _rechercher(args, ex)
        if nom == "ajouter_a_la_liste":
            return _ajouter(args, ex.state)
        return executer_tool(nom, args)
    except Exception as e:  # filet de sécurité : l'agent ne plante jamais à cause d'un outil
        return {"ok": False, "erreur": f"Erreur dans l'outil {nom} : {type(e).__name__}: {e}"}


def _lire_arguments(brut):
    """Les arguments arrivent en texte JSON (format OpenAI). Retourne (dict, message d'erreur ou None)."""
    if brut in (None, ""):
        return {}, None
    if isinstance(brut, dict):
        return brut, None
    try:
        valeur = json.loads(brut)
    except (TypeError, json.JSONDecodeError):
        return {}, f"Arguments illisibles (JSON invalide) : {str(brut)[:200]}"
    if not isinstance(valeur, dict):
        return {}, "Les arguments doivent être un objet JSON, ex. {\"nom\": \"lait de coco\"}."
    return valeur, None


# ---------------------------------------------------------------------------------------------- confirmation

_OUI_DEBUT = {"oui", "ouais", "yes", "ok", "okay", "d'accord", "je", "confirme", "confirmer", "vas", "go",
              "envoie", "envoies", "envoyer", "valide", "c'est", "parfait"}
_OUI_RESTE = _OUI_DEBUT | {"y", "la", "le", "les", "ma", "liste", "stp", "svp", "merci", "bon", "s'il", "te", "vous",
                           "plait", "tout", "de", "suite", "maintenant", "ca", "bien", "sur"}
_NON_DEBUT = {"non", "no", "annule", "annuler", "stop", "attends", "attend", "pas"}


def decision_confirmation(texte):
    """« oui » seulement pour une confirmation sans autre demande ; « non » pour un refus ; None sinon."""
    mots = re.findall(r"[a-z']+", norm(texte))
    if not mots:
        return None
    if mots[0] in _NON_DEBUT:
        return "non"
    if mots[0] in _OUI_DEBUT and all(m in _OUI_RESTE for m in mots):
        return "oui"
    return None


# ---------------------------------------------------------------------------------------------- boucle

def _system(st):
    texte = SYSTEM
    if st["shopping_list"]:
        texte += "\n\nListe de courses actuelle : " + " ; ".join(st["shopping_list"])
        if st["sent"]:
            texte += " (déjà envoyée)"
    return texte


def _sources_citees(texte, extraits):
    chunks = level1.get_index().chunks if extraits else []
    docs = []
    for groupe in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", texte):  # [1], [1, 2]
        for n in map(int, groupe.split(",")):
            if 1 <= n <= len(extraits):
                d = chunks[extraits[n - 1]]["doc_id"]
                if d not in docs:
                    docs.append(d)
    return docs


def _sortie(ex, question, texte, t0, seed, arret):
    st = ex.state
    if st["pending_confirmation"]:
        texte += (f"\n\nPour envoyer la liste ({len(st['articles'])} articles), réponds « oui ». "
                  "Réponds « non » pour annuler.")
    elif arret in ("max_steps", "max_tokens") and st["shopping_list"]:
        texte += "\n\nListe actuelle : " + " ; ".join(st["shopping_list"])
    st["historique"] = (st["historique"] + [{"role": "user", "content": question},
                                            {"role": "assistant", "content": texte}])[-HISTORIQUE:]
    return {
        "answer": texte,
        "sources": _sources_citees(texte, ex.extraits),
        "retrieved": ex.retrieved if ex.retrieved is not None else [],
        "state": st,
        "steps": ex.steps,
        "tokens_in": ex.tokens_in,
        "tokens_out": ex.tokens_out,
        "latency": time.time() - t0,
        "arret": arret,                    # reponse | max_steps | max_tokens | confirmation
        "seed": seed,
        "contexte_max": ex.contexte_max,   # plus gros prompt envoyé : à comparer à la fenêtre de contexte d'Ollama
    }


def warmup():
    """Hors chronomètre (eval.py) : index du RAG et chargement du modèle en mémoire."""
    level1.warmup()
    if llm.MODEL != "mock":
        try:
            llm.chat([{"role": "user", "content": "Bonjour"}], max_tokens=1)
        except Exception:
            pass


def answer(question: str, state: dict = None) -> dict:
    t0 = time.time()
    ex = _Execution(_etat(state))
    st = ex.state
    seed = random.randrange(1_000_000)

    # 1. Un envoi attend-il une confirmation ? C'est le code qui tranche, pas le modèle.
    if st["pending_confirmation"]:
        decision = decision_confirmation(question)
        if decision == "oui":
            res = _envoyer(st)
            ex.steps.append({"etape": 0, "type": "confirmation", "decision": "oui", "resultat": res})
            return _sortie(ex, question, f"C'est envoyé : ta liste de courses ({len(st['articles'])} articles) est partie.",
                           t0, seed, "confirmation")
        st["pending_confirmation"] = False  # tout autre message annule la demande d'envoi
        ex.steps.append({"etape": 0, "type": "confirmation", "decision": decision or "autre demande",
                         "resultat": "envoi annulé"})
        if decision == "non" and len(re.findall(r"[a-z']+", norm(question))) <= 3:
            return _sortie(ex, question, "D'accord, je n'envoie pas la liste. Tu peux encore la modifier.",
                           t0, seed, "confirmation")

    messages = [{"role": "system", "content": _system(st)}, *st["historique"], {"role": "user", "content": question}]

    # Variante : la recherche est lancée par le code avant le premier appel au modèle.
    if RECHERCHE_INITIALE:
        args = {"requete": question}
        res = _executer("rechercher_recettes", args, ex)
        appel = {"id": "recherche_initiale", "type": "function",
                 "function": {"name": "rechercher_recettes", "arguments": json.dumps(args, ensure_ascii=False)}}
        messages += [{"role": "assistant", "content": "", "tool_calls": [appel]},
                     {"role": "tool", "tool_call_id": "recherche_initiale", "content": json.dumps(res, ensure_ascii=False)}]
        ex.steps.append({"etape": 0, "type": "recherche_initiale", "outils": [
            {"outil": "rechercher_recettes", "effet": "lecture", "arguments": args, "resultat": _abreger(res)}]})

    # 2. La boucle de l'agent, bornée en nombre d'étapes et en tokens.
    texte, arret, relance = "", "max_steps", False
    for etape in range(1, MAX_STEPS + 1):
        if ex.tokens_in + ex.tokens_out >= MAX_TOKENS:
            arret = "max_tokens"
            break
        r = llm.chat(messages, tools=SCHEMAS, temperature=TEMPERATURE, max_tokens=MAX_SORTIE, seed=seed + etape)
        ex.tokens_in += r["tokens_in"]
        ex.tokens_out += r["tokens_out"]
        ex.contexte_max = max(ex.contexte_max, r["tokens_in"])
        msg = r["message"] or {}
        appels = msg.get("tool_calls") or []
        pas = {"etape": etape, "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
               "duree_llm": round(r["latency"], 2), "outils": []}
        ex.steps.append(pas)

        if not appels:  # le modèle répond sans outil : c'est la réponse finale
            texte = (msg.get("content") or "").strip()
            if not texte and not relance:  # réponse vide (déjà observée avec qwen2.5:3b) : on relance une fois
                relance = True
                pas["remarque"] = "réponse vide, relance"
                continue
            arret = "reponse"
            break

        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": appels})
        for i, appel in enumerate(appels):
            f = appel.get("function") or {}
            nom = f.get("name") or ""
            args, erreur = _lire_arguments(f.get("arguments"))
            t1 = time.time()
            res = {"ok": False, "erreur": erreur} if erreur else _executer(nom, args, ex)
            pas["outils"].append({"outil": nom, "effet": EFFETS.get(nom, "inconnu"), "arguments": args,
                                  "resultat": _abreger(res), "duree": round(time.time() - t1, 3)})
            messages.append({"role": "tool", "tool_call_id": appel.get("id") or f"appel_{etape}_{i}",
                             "content": json.dumps(res, ensure_ascii=False)})

    if arret == "max_steps":
        texte = f"Je n'ai pas pu terminer la demande en {MAX_STEPS} étapes."
    elif arret == "max_tokens":
        texte = f"Je n'ai pas pu terminer la demande dans la limite de {MAX_TOKENS} tokens."
    return _sortie(ex, question, texte, t0, seed, arret)
