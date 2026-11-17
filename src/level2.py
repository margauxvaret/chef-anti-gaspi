"""Niveau 2 : agent outillé. La boucle est écrite à la main, sans framework.

Le modèle choisit les outils ; le CODE les exécute, applique les règles métier et garde la main sur
l'action irréversible (l'envoi de la liste), qui n'est jamais exécutée sans un « oui » de l'utilisateur.

Outils, classés par effet (dictionnaire EFFETS) :
    rechercher_recettes   lecture        extraits du livre de cuisine (RAG du niveau 1), cités [n]
    convertir_unites      lecture        src/tools.py
    chercher_produit      lecture        src/tools.py (Open Food Facts)
    ajouter_recette       écriture       le CODE lit les ingrédients d'une recette du livre (src/recettes.py),
                                         retire ce que l'utilisateur a déjà, fait la règle de trois
    ajouter_a_la_liste    écriture       articles isolés (ex. un produit trouvé dans Open Food Facts)
    envoyer_liste         irréversible   jamais exécuté par le modèle : met l'envoi en attente de confirmation

Contrat commun : answer(question) -> dict (voir src/level0.py). En plus, answer(question, state) reprend
l'état renvoyé au tour précédent : c'est ce qui permet de répondre « oui » pour confirmer un envoi
(interface Streamlit). Sans état, chaque appel repart de zéro, ce dont eval.py a besoin.

Réglages (variables d'environnement) :
    LEVEL2_MAX_STEPS           appels au modèle par question (défaut 8)
    LEVEL2_MAX_TOKENS          tokens cumulés, entrée + sortie, par question (défaut 20000)
    LEVEL2_TEMPERATURE         défaut 0.3, avec une seed tirée au hasard (notée dans la sortie) : à 0, les 3 essais
                               de pass^3 seraient identiques ; à 0.7, qwen2.5:3b divaguait (run du 16/11)
    LEVEL2_RELANCE             1 (défaut) = si le modèle répond en texte alors que la tâche n'est pas faite (liste
                               demandée mais vide, envoi demandé mais pas préparé, « je vais… »), le code le relance :
                               2 fois au plus par motif, 3 en tout. 0 = désactivé (pour mesurer l'effet des relances)

Règles métier tenues par le code (et non par le prompt), voir _ajouter :
    - quantités positives ; unités métriques ramenées en g et ml ; doublons regroupés ;
    - ingrédients lus par le code dans le livre, règle de trois faite par le code, pas par le modèle ;
    - ancrage dans la demande : « sauf » et « seulement » ne gardent que ce que l'utilisateur a écrit ; une recette
      non demandée est refusée quand l'utilisateur en a cité ; un ingrédient d'une recette citée passe par
      ajouter_recette (quantités du livre) et non par ajouter_a_la_liste (quantités inventées) ;
    - allergies déclarées (« je suis allergique aux arachides ») : les articles concernés sont refusés ;
    - pas d'envoi d'une liste vide ; liste modifiée = confirmation annulée ; appel identique refusé.
    LEVEL2_TOP_K               extraits renvoyés par recherche (défaut 3 ; le contexte d'Ollama est limité)
    LEVEL2_RECHERCHE_INITIALE  1 = le code lance la recherche sur la question avant le premier appel au modèle
                               (variante à mesurer contre 0, le défaut, où le modèle décide seul)
"""
import copy
import json
import math
import os
import random
import re
import time
import unicodedata
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from src import level1, llm
from src import recettes as R
from src.tools import SCHEMAS as SCHEMAS_TOOLS
from src.tools import executer as executer_tool

ROOT = Path(__file__).resolve().parent.parent
ENVOIS = ROOT / "runs" / "envois.jsonl"  # l'envoi est simulé : chaque liste envoyée est ajoutée à ce journal

MAX_STEPS = int(os.getenv("LEVEL2_MAX_STEPS", "8"))
MAX_TOKENS = int(os.getenv("LEVEL2_MAX_TOKENS", "20000"))
TEMPERATURE = float(os.getenv("LEVEL2_TEMPERATURE", "0.3"))
TOP_K = int(os.getenv("LEVEL2_TOP_K", "3"))
RECHERCHE_INITIALE = os.getenv("LEVEL2_RECHERCHE_INITIALE", "0") == "1"
RELANCE = os.getenv("LEVEL2_RELANCE", "1") == "1"
MAX_RELANCES = 3       # en tout par question
MAX_PAR_MOTIF = 2      # par motif de relance
MAX_EXTRAITS_RECETTE = 4
MAX_SORTIE = 1024     # tokens par réponse du modèle : une liste de 20 articles en JSON dépasse 512
MAX_ARTICLES = 60
HISTORIQUE = 6        # messages (questions et réponses) gardés d'un tour à l'autre

ABSTAIN = level1.ABSTAIN

EFFETS = {
    "rechercher_recettes": "lecture",
    "convertir_unites": "lecture",
    "chercher_produit": "lecture",
    "ajouter_recette": "ecriture",
    "ajouter_a_la_liste": "ecriture",
    "envoyer_liste": "irreversible",
}


def _schema(nom, description, proprietes, requis):
    return {"type": "function", "function": {"name": nom, "description": description, "parameters": {
        "type": "object", "properties": proprietes, "required": requis}}}


def _outils_de_base():
    """Schémas de src/tools.py. La description de chercher_produit est resserrée : celle d'origine (« utile pour
    choisir un produit à ajouter à la liste ») poussait qwen2.5:3b à l'appeler pour chaque ingrédient."""
    schemas = copy.deepcopy(SCHEMAS_TOOLS)
    for s in schemas:
        if s["function"]["name"] == "chercher_produit":
            s["function"]["description"] = (
                "Cherche un produit du commerce dans Open Food Facts (marque, quantité, Nutri-Score). Seulement si "
                "l'utilisateur parle d'un produit du commerce, d'une marque, du Nutri-Score ou d'Open Food Facts.")
    return schemas


SCHEMAS = [
    _schema("rechercher_recettes",
            "Cherche dans le livre de cuisine les extraits utiles : ingrédients, quantités, temps, températures. "
            "Chaque extrait a un numéro [n] à citer.",
            {"requete": {"type": "string", "description": "Plat ou ingrédients recherchés, ex. pipérade"}}, ["requete"]),
    *_outils_de_base(),
    _schema("ajouter_recette",
            "Ajoute à la liste de courses les ingrédients d'une recette du livre. Le système lit lui-même les "
            "ingrédients et fait la règle de trois : ne recopie pas les ingrédients.",
            {"recette": {"type": "string", "description": "nom du plat, ex. pipérade"},
             "personnes": {"type": "integer", "description": "nombre de personnes demandé par l'utilisateur"},
             "sauf": {"type": "array", "items": {"type": "string"},
                      "description": "ce que l'utilisateur a déjà ou ne veut pas, ex. [\"oignons\", \"ail\"]"},
             "seulement": {"type": "array", "items": {"type": "string"},
                           "description": "si l'utilisateur ne veut que certains ingrédients, ex. [\"vin blanc\"]"}},
            ["recette"]),
    _schema("ajouter_a_la_liste",
            "Ajoute des articles isolés à la liste de courses (ex. un produit trouvé dans Open Food Facts). "
            "Pour les ingrédients d'une recette du livre, utiliser ajouter_recette.",
            {"articles": {"type": "array", "items": {"type": "object", "properties": {
                "nom": {"type": "string", "description": "ex. oignons"},
                "quantite": {"type": "number", "description": "quantité écrite dans la recette, ex. 3 (facultatif)"},
                "unite": {"type": "string", "description": "ex. g, cl, gousse ; rien pour un nombre de pièces"}},
                "required": ["nom"]}},
             "personnes_recette": {"type": "integer", "description": "nombre de personnes prévu par la recette"},
             "personnes": {"type": "integer", "description": "nombre de personnes demandé par l'utilisateur"}},
            ["articles"]),
    _schema("envoyer_liste",
            "Envoie la liste de courses. Action irréversible : le système demande lui-même la confirmation "
            "à l'utilisateur.", {}, []),
]
assert {s["function"]["name"] for s in SCHEMAS} == set(EFFETS), "chaque outil doit avoir un effet déclaré"

SYSTEM = f"""Tu es un chef anti-gaspi. Réponds en français, en peu de mots. Agis avec les outils au lieu de décrire ce que tu vas faire.

QUESTION sur une recette : appelle rechercher_recettes, puis réponds avec les extraits en citant [n]. Si aucun extrait ne parle du plat demandé, réponds exactement : « {ABSTAIN} »

LISTE DE COURSES :
1) ajouter_recette avec recette = le nom du plat, personnes = le nombre demandé, sauf = ce que l'utilisateur a déjà ou ne veut pas, seulement = les ingrédients voulus s'il n'en veut que certains. Le système lit les ingrédients dans le livre et fait les calculs : ne recopie pas les ingrédients toi-même. Les allergies déclarées sont appliquées par le système.
2) Si la recette n'est pas dans le livre, n'invente pas ses ingrédients : applique la solution de repli de l'utilisateur s'il en a donné une, sinon dis-le.
3) Si l'utilisateur demande l'envoi : envoyer_liste. Le système lui demande la confirmation ; n'écris jamais que la liste est envoyée.
4) Une réponse courte qui résume la liste.
Pour un produit isolé (ex. un lait de coco trouvé dans Open Food Facts) : ajouter_a_la_liste.

Conversion d'unités : convertir_unites. Produit du commerce, marque ou Nutri-Score : chercher_produit.
Demande trop vague : une question de précision. Demande sans rapport avec la cuisine : refus poli.
Si un outil renvoie une erreur, corrige ton appel."""


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


# Unités métriques ramenées à g et ml : « 20 cl » et « 100 ml » de vin blanc se regroupent en une ligne.
_VOLUMES = {"ml": 1, "millilitre": 1, "cl": 10, "centilitre": 10, "dl": 100, "l": 1000, "litre": 1000}
_MASSES = {"g": 1, "gr": 1, "gramme": 1, "kg": 1000, "kilo": 1000, "kilogramme": 1000, "mg": 0.001}
_SANS_UNITE = {"unite", "piece", "u", "pc", "pcs", "nombre", "un", "une"}


def _normaliser_unite(q, unite):
    """(quantité, unité) -> (quantité, unité) en g ou ml quand c'est possible ; les autres unités restent."""
    u = norm(unite).strip().rstrip(".")
    u_sing = u[:-1] if len(u) > 2 and u.endswith("s") else u
    if u_sing in _SANS_UNITE or u in _SANS_UNITE:
        return q, ""
    for table, cible in ((_VOLUMES, "ml"), (_MASSES, "g")):
        facteur = table.get(u, table.get(u_sing))
        if facteur is not None:
            return (None if q is None else q * facteur), cible
    return q, unite


def _arrondir(q, unite):
    """Après une règle de trois : on achète des pièces entières ; g et ml arrondis à l'unité."""
    if unite in ("g", "ml"):
        return float(round(q)) if q >= 10 else round(q, 1)
    return float(math.ceil(q - 1e-9))


# Allergies : un mot déclaré par l'utilisateur -> les mots qui le désignent dans un nom d'ingrédient.
ALLERGENES = {
    "arachide": ["arachide", "cacahuete", "cacahouete"],
    "fruits a coque": ["noix", "noisette", "amande", "pistache", "cajou", "pecan", "macadamia"],
    "lait": ["lait", "beurre", "creme", "fromage", "yaourt", "mozzarella", "emmental", "parmesan", "gruyere"],
    "lactose": ["lait", "beurre", "creme", "fromage", "yaourt", "mozzarella", "emmental", "parmesan", "gruyere"],
    "oeuf": ["oeuf"],
    "poisson": ["poisson", "thon", "saumon", "anchois", "cabillaud", "sardine", "maquereau", "nam pla", "nuoc mam"],
    "crustace": ["crevette", "crabe", "homard", "langoustine", "ecrevisse"],
    "gluten": ["ble", "farine", "pain", "chapelure", "semoule", "seigle", "orge", "epeautre", "biscuit"],
    "soja": ["soja"],
    "sesame": ["sesame"],
    "moutarde": ["moutarde"],
    "celeri": ["celeri"],
}
_ALLERGIE = re.compile(r"allergi(?:que|e)s?\s+(?:aux|au|a la|a l'|a|d'|de|des)\s*([^.;!?\n]+)")
_PREPOSITIONS = re.compile(r"^(?:aux|au|a la|a l'|a|d'|de|des|du|la|le|les)\s+")


def allergies_declarees(texte):
    """« Je suis allergique aux arachides et au lait » -> ["arachide", "lait"].
    Un allergène hors de la table est gardé s'il tient en un mot (ex. « kiwi »)."""
    trouvees = []
    for groupe in _ALLERGIE.findall(norm(texte)):
        for partie in re.split(r",|\bet\b|\bou\b", groupe):
            mot = _cle(_PREPOSITIONS.sub("", partie.strip()))
            if not mot:
                continue
            cle = next((k for k in ALLERGENES if _cle(k) == mot), None)
            if cle is None and len(mot.split()) != 1:
                continue  # morceau de phrase (« prepare la liste »), pas un allergène
            cle = cle or mot
            if cle not in trouvees:
                trouvees.append(cle)
    return trouvees


def _allergene(nom, allergies):
    """Premier allergène déclaré présent dans le nom (mots entiers : « lait » ne bloque pas « laitue »)."""
    n = f" {_cle(nom)} "
    for a in allergies:
        if any(f" {_cle(m)} " in n for m in ALLERGENES.get(a, [a])):
            return a
    return None


# « J'ai déjà la viande et les épices » : un mot générique couvre plusieurs ingrédients.
CATEGORIES = {
    "viande": ["viande", "boeuf", "veau", "porc", "agneau", "mouton", "poulet", "dinde", "canard", "lapin", "paleron",
               "basse cote", "steak", "jambon", "lardon", "saucisse", "chorizo", "bacon", "gigot", "collier"],
    "epice": ["epice", "cumin", "paprika", "origan", "curry", "curcuma", "cannelle", "muscade", "gingembre", "piment",
              "poivre", "safran", "cardamome", "ras el hanout", "chili", "cayenne", "girofle"],
    "herbe": ["herbe", "persil", "basilic", "coriandre", "ciboulette", "thym", "laurier", "romarin", "menthe", "aneth",
              "estragon", "sauge"],
    "poisson": ALLERGENES["poisson"],
}


def _sans_article(terme):
    """« de la farine » -> « farine »"""
    t = norm(terme).strip()
    while True:
        court = re.sub(r"^(?:(?:les|le|la|des|du|de)\s+|(?:l'|d')\s*)", "", t)
        if court == t:
            return t
        t = court


def _developper(terme):
    """« les épices » -> les ingrédients de la catégorie ; sinon le terme lui-même."""
    t = _sans_article(terme)
    return CATEGORIES.get(R.cle(t), [t])


def _textes_utilisateur(ex):
    return [ex.question] + [m["content"] for m in ex.state["historique"] if m["role"] == "user"]


def _ancrer(termes, ex):
    """Ne garde que les termes que l'utilisateur a vraiment écrits. Retourne (gardés, ignorés)."""
    textes = _textes_utilisateur(ex)
    gardes = [t for t in termes if any(R.contient(x, _sans_article(t)) for x in textes)]
    return gardes, [t for t in termes if t not in gardes]


def _textes(valeur):
    """Liste de textes, que le modèle envoie un tableau ou « oignons, ail et persil »."""
    if valeur in (None, ""):
        return []
    if isinstance(valeur, str):
        valeur = re.split(r",|\bet\b", valeur)
    if not isinstance(valeur, list):
        valeur = [valeur]
    return [str(v).strip() for v in valeur if str(v).strip()]


_MOTS_VIDES = {"pour", "personne", "personnes", "recette", "recettes", "avec", "dans", "sans", "liste", "course",
               "courses", "ingredient", "ingredients", "quantite", "faire", "comment", "plat"}


def titre_correspond(requete, titre):
    """La requête nomme-t-elle cette recette ? (au moins un mot significatif en commun avec le titre)"""
    mots = {m for m in _cle(requete).split() if len(m) >= 4 and m not in _MOTS_VIDES}
    return bool(mots & set(_cle(titre).split())) if mots else True


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
        self.liste_modifiee = False
        self.envoi_prepare = False
        self.demande_liste = False
        self.question = ""
        self.produits_cherches = []   # noms passés à chercher_produit (exception à la redirection vers la recette)
        self.relances = 0
        self.relances_par_motif = {}


def _etat(state):
    st = copy.deepcopy(state) if state else {}
    st.setdefault("articles", [])
    st.setdefault("sent", False)
    st.setdefault("pending_confirmation", False)
    st.setdefault("historique", [])
    st.setdefault("allergies", [])
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
                "remarque": f"Aucune recette trouvée pour « {requete} ». Pour une question, réponds « {ABSTAIN} » ; "
                            "pour une liste de courses, dis que ce plat n'est pas dans le livre, ou suis la solution "
                            "de repli proposée par l'utilisateur."}
    chunks = level1.get_index().chunks
    if ex.demande_liste:
        return _recette_complete(requete, ranked, chunks, ex)
    extraits = []
    for i in ranked[:TOP_K]:
        if i not in ex.extraits:
            ex.extraits.append(i)
        extraits.append({"n": ex.extraits.index(i) + 1, "recette": level1.short_title(chunks[i]["title"]),
                         "texte": chunks[i]["text"]})
    recettes = list(dict.fromkeys(e["recette"] for e in extraits))
    return {"ok": True, "recettes_trouvees": recettes, "extraits": extraits,
            "suite": "Pour une question : réponds avec ces extraits en citant [n]. Pour une liste de courses : "
                     "appelle maintenant ajouter_a_la_liste avec les ingrédients de la bonne recette."}


_PERSONNES = re.compile(r"pour\s+(\d+)\s+personnes", re.I)


def _recette_complete(requete, ranked, chunks, ex):
    """Pour une liste de courses : UNE recette entière (la mieux classée) plutôt que des extraits de recettes
    différentes, que qwen2.5:3b mélangeait (run du 17/11 : semoule et harissa dans le chili con carne)."""
    docs = level1.doc_ranking(ranked)
    meilleur = docs[0]
    ids = [i for i, c in enumerate(chunks) if c["doc_id"] == meilleur]
    if len(ids) > MAX_EXTRAITS_RECETTE:  # longue recette : le début (ingrédients) et les extraits les mieux classés
        rang = {i: r for r, i in enumerate(ranked)}
        ids = sorted([ids[0]] + sorted(ids[1:], key=lambda i: rang.get(i, 10 ** 6))[:MAX_EXTRAITS_RECETTE - 1])
    extraits = []
    for i in ids:
        if i not in ex.extraits:
            ex.extraits.append(i)
        extraits.append({"n": ex.extraits.index(i) + 1, "texte": chunks[i]["text"]})
    titre = level1.short_title(chunks[ids[0]]["title"])
    m = _PERSONNES.search(" ".join(chunks[i]["text"] for i in ids))
    res = {"ok": True, "recette": titre, "personnes_recette": int(m.group(1)) if m else None, "extraits": extraits,
           "autres_recettes_proches": [level1.short_title(next(c["title"] for c in chunks if c["doc_id"] == d))
                                       for d in docs[1:4]]}
    if not titre_correspond(requete, titre):
        res["attention"] = (f"Aucune recette du livre ne s'appelle « {requete} ». Ne prends pas « {titre} » à la "
                            "place : applique la solution de repli de l'utilisateur s'il en a donné une, sinon "
                            "dis-lui que ce plat n'est pas dans le livre.")
    res["suite"] = (f"Pour la liste de courses, appelle maintenant ajouter_recette(recette=\"{titre}\", personnes, "
                    "sauf) : le système lit les ingrédients.")
    return res


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

    facteur = None
    pr, pv = args.get("personnes_recette"), args.get("personnes")
    if pr not in (None, "") and pv not in (None, ""):
        try:
            pr, pv = int(_nombre(pr)), int(_nombre(pv))
        except ValueError:
            return {"ok": False, "erreur": "personnes_recette et personnes doivent être des nombres entiers."}
        if not (1 <= pr <= 100 and 1 <= pv <= 100):
            return {"ok": False, "erreur": "personnes_recette et personnes doivent être entre 1 et 100."}
        facteur = pv / pr if pv != pr else None

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
        allergene = _allergene(nom, st.get("allergies", []))
        if allergene:
            refuses.append({"article": nom, "raison": f"allergie déclarée : {allergene}"})
            continue
        q, unite = _normaliser_unite(q, str(a.get("unite") or "").strip())
        if facteur and q is not None:
            q = _arrondir(q * facteur, unite)
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
    if facteur:
        res["regle_de_trois"] = f"quantités multipliées par {_fmt(round(facteur, 3))} ({pr} -> {pv} personnes)"
    if refuses:
        res["refuses"] = refuses
    if not ajoutes:
        res["erreur"] = "Aucun article ajouté : voir « refuses »."
    if remarques:
        res["remarques"] = remarques
    return res


def _ajouter_recette(args, ex):
    nom = str(args.get("recette") or "").strip()
    doc = R.trouver_recette(nom)
    # recettes du livre citées dans la conversation (l'utilisateur, ou l'assistant à un tour précédent)
    citees = R.recettes_citees(_textes_utilisateur(ex) + [m["content"] for m in ex.state["historique"]])
    titres_cites = ", ".join(R.recettes()[d]["titre"] for d in citees)
    if not doc:
        ranked, _ = level1.retrieve(nom) if nom else ([], None)
        proches = [R.recettes()[d]["titre"] for d in level1.doc_ranking(ranked)[:3] if d in R.recettes()]
        return {"ok": False, "erreur": f"Pas de recette « {nom} » dans le livre (recettes proches : "
                f"{', '.join(proches) or 'aucune'}). N'invente pas ses ingrédients : applique la solution de repli "
                "de l'utilisateur s'il en a donné une, sinon dis-lui que ce plat n'est pas dans le livre."
                + (f" Recettes du livre citées par l'utilisateur : {titres_cites}." if citees else "")}
    if citees and doc not in citees:  # run du 17/11 (multi-10) : la pizza proposée par la recherche au lieu de la pipérade
        return {"ok": False, "erreur": f"L'utilisateur n'a pas demandé « {R.recettes()[doc]['titre']} ». Recettes du "
                f"livre citées dans sa demande : {titres_cites}. Si le plat demandé n'est pas dans le livre, applique "
                "sa solution de repli."}
    titre = R.recettes()[doc]["titre"]
    personnes_recette, articles = R.lire_ingredients(R.recettes()[doc]["texte"])
    if len(articles) < 2:
        return {"ok": False, "erreur": f"Les ingrédients de « {titre} » ne sont pas lisibles automatiquement : "
                                       "utilise rechercher_recettes puis ajouter_a_la_liste."}
    tous = [a["nom"] for a in articles]
    # run du 17/11 (multi-02, multi-03) : le modèle retirait des ingrédients que l'utilisateur n'avait pas cités
    seulement, seulement_ignores = _ancrer(_textes(args.get("seulement")), ex)
    sauf, sauf_ignores = _ancrer(_textes(args.get("sauf")), ex)
    if seulement:
        articles = [a for a in articles if any(R.contient(a["nom"], t) for t in seulement)]
        if not articles:
            return {"ok": False, "erreur": f"Aucun ingrédient « {', '.join(seulement)} » dans « {titre} ». "
                                           f"Ingrédients : {', '.join(tous)}."}
    retires = []
    termes = [m for t in sauf for m in _developper(t)]
    if termes:
        retires = [a["nom"] for a in articles if any(R.contient(a["nom"], m) for m in termes)]
        articles = [a for a in articles if a["nom"] not in retires]
    remarques = []
    demande = {"articles": articles}
    if args.get("personnes") not in (None, ""):
        if personnes_recette:
            demande.update(personnes_recette=personnes_recette, personnes=args["personnes"])
        else:
            remarques.append("La recette n'indique pas pour combien de personnes : quantités de la recette, non adaptées.")
    if not articles:
        res = {"ok": True, "ajoutes": [], "liste": ex.state["shopping_list"],
               "remarque": "Rien à ajouter : l'utilisateur a déjà tous les ingrédients."}
    else:
        res = _ajouter(demande, ex.state)
    res.update(recette=titre, personnes_recette=personnes_recette)
    if retires:
        res["retires"] = retires
    if sauf_ignores or seulement_ignores:
        res["ignores"] = {"sauf": sauf_ignores, "seulement": seulement_ignores,
                          "raison": "non mentionnés par l'utilisateur"}
    if remarques:
        res["remarques"] = res.get("remarques", []) + remarques
    return res


def _redirection_vers_recette(args, ex):
    """Un article d'une recette citée par l'utilisateur doit venir du livre (run du 17/11, multi-04 et multi-08 :
    « vin blanc : 100 ml » et « oignons : 2 g » inventés). Retourne l'erreur à renvoyer au modèle, ou None."""
    citees = R.recettes_citees(_textes_utilisateur(ex))
    articles = args.get("articles")
    if isinstance(articles, str):
        try:
            articles = json.loads(articles)
        except json.JSONDecodeError:
            return None
    if isinstance(articles, dict):
        articles = [articles]
    if not citees or not isinstance(articles, list):
        return None
    noms = [str(a.get("nom") if isinstance(a, dict) else a) for a in articles]
    noms = [n for n in noms if not any(R.contient(n, p) or R.contient(p, n) for p in ex.produits_cherches)]
    for doc in citees:
        ingredients = [i["nom"] for i in R.lire_ingredients(R.recettes()[doc]["texte"])[1]]
        venus = [n for n in noms if any(R.contient(n, i) or R.contient(i, n) for i in ingredients)]
        if venus:
            titre = R.recettes()[doc]["titre"]
            return {"ok": False, "erreur": f"{', '.join(venus)} : ingrédient(s) de la recette « {titre} » du livre. "
                    f"Utilise ajouter_recette(recette=\"{titre}\", personnes, seulement={json.dumps(venus, ensure_ascii=False)}) "
                    "pour avoir les quantités du livre."}
    return None


def _demander_confirmation(st):
    """Ce que reçoit le modèle quand il appelle envoyer_liste : rien n'est envoyé."""
    if not st["articles"]:
        return {"ok": False, "erreur": "La liste de courses est vide : il n'y a rien à envoyer."}
    if st["sent"]:
        return {"ok": False, "erreur": "Cette liste a déjà été envoyée."}
    if st["pending_confirmation"]:
        return {"ok": True, "statut": "deja_en_attente", "detail": "envoi déjà en attente de la confirmation"}
    st["pending_confirmation"] = True
    return {"ok": True, "statut": "en_attente_de_confirmation", "liste": st["shopping_list"],
            "detail": "rien n'est envoyé : le système demande la confirmation à l'utilisateur"}


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
    effet = EFFETS.get(nom)
    if effet is None:
        return {"ok": False, "erreur": f"Outil inconnu : {nom!r}. Outils disponibles : {', '.join(EFFETS)}."}
    try:
        if effet == "irreversible":  # règle générale : une action irréversible attend toujours un « oui »
            res = _demander_confirmation(ex.state)
            ex.envoi_prepare = ex.envoi_prepare or bool(res.get("ok"))
        elif nom == "rechercher_recettes":
            res = _rechercher(args, ex)
        elif nom == "ajouter_recette":
            res = _ajouter_recette(args, ex)
            ex.liste_modifiee = ex.liste_modifiee or bool(res.get("ok"))
        elif nom == "ajouter_a_la_liste":
            res = _redirection_vers_recette(args, ex) or _ajouter(args, ex.state)
            ex.liste_modifiee = ex.liste_modifiee or bool(res.get("ok"))
        else:
            res = executer_tool(nom, args)
            if nom == "chercher_produit" and res.get("ok") and args.get("nom"):
                ex.produits_cherches.append(str(args["nom"]))
    except Exception as e:  # filet de sécurité : l'agent ne plante jamais à cause d'un outil
        res = {"ok": False, "erreur": f"Erreur dans l'outil {nom} : {type(e).__name__}: {e}"}
    # On ne mémorise que les appels réussis : un appel qui a échoué (liste encore vide, etc.) peut être refait.
    # L'envoi n'est jamais mémorisé : son résultat dépend de l'état (run du 17/11, multi-09).
    if res.get("ok") and effet != "irreversible":
        ex.deja_vus.add(cle)
    return res


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


# ---------------------------------------------------------------------------------------------- relances

_ANNONCE = re.compile(r"\b(je vais|nous allons|on va|ajoutons|commencons par)\b")
_ENVOI = re.compile(r"\b(envoie|envoies|envoyer)\b")
_PAS_ENVOI = re.compile(r"n'envoie|ne l'envoie|sans l'envoyer|pas l'envoyer|ne pas envoyer")


def demande_liste(question):
    q = norm(question)
    return "liste" in q or "courses" in q


def consigne_de_relance(question, texte, ex):
    """Le modèle répond en texte : la tâche demandée est-elle faite ? Sinon, (motif, consigne à lui renvoyer).

    Le code ne fait rien à la place du modèle : il lui rappelle l'étape qui manque. L'envoi reste soumis
    à la confirmation de l'utilisateur, relance ou pas."""
    q, t = norm(question), norm(texte)
    regles = []
    if ex.demande_liste and not ex.liste_modifiee:
        regles.append(("liste_vide", "Tu n'as encore rien ajouté à la liste de courses. Appelle maintenant "
                       "ajouter_recette (recette, personnes, sauf) ou, pour un produit isolé, ajouter_a_la_liste. "
                       "Ne réponds pas en texte."))
    if ex.demande_liste and ex.liste_modifiee and _ENVOI.search(q) and not _PAS_ENVOI.search(q) and not ex.envoi_prepare:
        regles.append(("envoi", "L'utilisateur a demandé l'envoi : appelle maintenant envoyer_liste "
                       "(le système lui demandera de confirmer)."))
    if _ANNONCE.search(t) and "?" not in texte:
        regles.append(("annonce", "Ne décris pas ce que tu vas faire : appelle directement l'outil nécessaire, "
                       "ou donne ta réponse finale."))
    for motif, consigne in regles:
        if ex.relances_par_motif.get(motif, 0) < MAX_PAR_MOTIF:
            return motif, consigne
    return None


# ---------------------------------------------------------------------------------------------- boucle

def _system(st):
    texte = SYSTEM
    if st["allergies"]:
        texte += "\n\nAllergies déclarées par l'utilisateur : " + ", ".join(st["allergies"])
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
        "relances": ex.relances,
        "relances_par_motif": ex.relances_par_motif,
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

    for a in allergies_declarees(question):  # gardées dans l'état pour les tours suivants
        if a not in st["allergies"]:
            st["allergies"].append(a)
    ex.demande_liste = demande_liste(question)
    ex.question = question

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
            relance_ = None
            if RELANCE and ex.relances < MAX_RELANCES and etape < MAX_STEPS:
                relance_ = consigne_de_relance(question, texte, ex)
            if relance_:  # la tâche n'est pas faite : on rappelle au modèle l'étape qui manque
                motif, consigne = relance_
                ex.relances += 1
                ex.relances_par_motif[motif] = ex.relances_par_motif.get(motif, 0) + 1
                pas["relance"] = consigne
                messages += [{"role": "assistant", "content": texte}, {"role": "user", "content": consigne}]
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

    if arret == "reponse" and ex.liste_modifiee and ABSTAIN.rstrip(".") in texte:
        texte = ""  # « je ne trouve pas » alors que la liste a été remplie : la réponse contredit l'état
    if arret == "reponse" and not texte:  # réponse vide : on répond à partir de l'état
        texte = ("Ta liste de courses : " + " ; ".join(st["shopping_list"])) if st["shopping_list"] else \
            "Je n'ai pas réussi à répondre, peux-tu reformuler ?"
    if arret == "max_steps":
        texte = f"Je n'ai pas pu terminer la demande en {MAX_STEPS} étapes."
    elif arret == "max_tokens":
        texte = f"Je n'ai pas pu terminer la demande dans la limite de {MAX_TOKENS} tokens."
    return _sortie(ex, question, texte, t0, seed, arret)
