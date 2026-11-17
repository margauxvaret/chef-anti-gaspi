"""Lecture déterministe des recettes du livre : trouver une recette par son nom, lire ses ingrédients.

Utilisé par l'outil ajouter_recette du niveau 2 : le modèle dit QUELLE recette, pour COMBIEN de personnes et
SANS QUOI ; le code lit la section « Ingrédients », fait la règle de trois et remplit la liste. qwen2.5:3b ne
savait pas recopier une liste d'ingrédients en JSON (run du 17/11 : une seule ligne pour la pipérade,
« oignons : 80 ml » pour le chili).

Le lecteur couvre les formes courantes du livre :
    10 œufs | 1 kg de paleron | 20 cl de vin blanc sec, | 2 gousses d'ail | 16 cuillères à soupe huile d'olive
    Beurre (100 grammes) | 1 oignon et 1 gousse d'ail | Huile d'olive, sel, poivre | Quatre bananes plantains
Ce qu'il ne comprend pas reste un article sans quantité (le nom est gardé).
"""
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def norm(s):
    s = unicodedata.normalize("NFD", str(s or "").lower().replace("’", "'").replace("œ", "oe").replace("æ", "ae"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def cle(nom):
    """« Pommes de terre » -> « pomme de terre » : mots sans accents, au singulier approximatif."""
    return " ".join(m[:-1] if len(m) > 3 and m[-1] in "sx" else m for m in re.findall(r"[a-z0-9']+", norm(nom)))


def _mots(s):
    return cle(norm(s).replace("'", " "))


def contient(nom, terme):
    """Le terme apparaît-il en mots entiers dans le nom ? (« ail » dans « gousse d'ail », pas dans « volaille »)"""
    t = _mots(terme)
    return bool(t) and f" {t} " in f" {_mots(nom)} "


# ---------------------------------------------------------------------------------------------- recettes

@lru_cache(maxsize=1)
def recettes():
    """{doc_id: {"titre", "texte"}} pour tout le livre."""
    out = {}
    for f in sorted(CORPUS.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        titre = d.get("title", "").split("/")[-1].strip() or d["id"]
        out[d["id"]] = {"titre": titre, "texte": d.get("text", "")}
    return out


_MOTS_VIDES = {"pour", "personne", "recette", "avec", "dans", "sans", "liste", "course", "ingredient", "quantite",
               "faire", "plat", "livre", "cuisine", "maison"}


def mots_significatifs(texte):
    return {m for m in _mots(texte).split() if len(m) >= 4 and m not in _MOTS_VIDES}


def recettes_citees(textes):
    """Recettes du livre dont tous les mots significatifs du titre apparaissent dans ces textes
    (« prends plutôt la pipérade » cite la Pipérade)."""
    mots = set()
    for t in textes:
        mots |= set(_mots(t).split())
    return [d for d, r in recettes().items() if mots_significatifs(r["titre"]) and mots_significatifs(r["titre"]) <= mots]


def trouver_recette(nom):
    """doc_id de la recette dont le TITRE contient le plus de mots du nom demandé, ou None."""
    demande = mots_significatifs(nom)
    if not demande:
        return None
    candidats = []
    for doc_id, r in recettes().items():
        titre = set(_mots(r["titre"]).split())
        score = len(demande & titre)
        if score:
            candidats.append((-score, len(titre), doc_id))  # plus de mots communs, puis titre le plus court
    return min(candidats)[2] if candidats else None


# ---------------------------------------------------------------------------------------------- ingrédients

_DEBUT = re.compile(r"^\s*ingr[ée]dients?\b(?!\s+facultatifs?)", re.I)
_FIN = re.compile(r"^\s*(pr[ée]paration|mat[ée]riel|ingr[ée]dients?\s+facultatifs?|instructions?|[ée]tapes?|r[ée]alisation|"
                  r"cuisson|m[ée]thode|service|variantes?|conseils?|astuces?|d[ée]roulement|recette)\b", re.I)
_PERSONNES = re.compile(r"pour\s+(\d+)(?:\s*(?:à|a|-)\s*(\d+))?\s+personnes", re.I)

_MOTS_NOMBRES = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8,
                 "neuf": 9, "dix": 10, "douze": 12, "quinze": 15, "vingt": 20}
_NOMBRE = r"(?:\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?|½|¼|¾)"
_QUANTITE = re.compile(rf"^\s*({_NOMBRE})(?:\s*(?:à|a|-|ou)\s*({_NOMBRE}))?\s*(.*)$", re.I)

# unités reconnues (après norm), des plus longues aux plus courtes
_UNITES = sorted([
    "cuilleres a soupe", "cuillere a soupe", "cuillerees a soupe", "cuilleree a soupe", "c. a soupe", "c. a s.",
    "cuilleres a cafe", "cuillere a cafe", "cuillerees a cafe", "cuilleree a cafe", "c. a cafe", "c. a c.",
    "grosses cuillerees a soupe", "grosse cuilleree a soupe",
    "kilogrammes", "kilogramme", "kilos", "kilo", "kg", "grammes", "gramme", "gr", "g", "mg",
    "litres", "litre", "l", "dl", "cl", "ml", "pincees", "pincee", "gousses", "gousse", "sachets", "sachet",
    "tranches", "tranche", "filets", "filet", "verres", "verre", "bols", "bol", "bottes", "botte", "brins", "brin",
    "feuilles", "feuille", "morceaux", "morceau", "tasses", "tasse", "poignees", "poignee", "bouquets", "bouquet",
    "tetes", "tete", "branches", "branche", "boites", "boite", "petites boites", "petite boite", "grandes boites",
    "grande boite", "pots", "pot", "paquets", "paquet", "noix", "noisettes", "quart de litre", "demi-litre",
    "c.c", "c.s", "cc", "cs",
], key=len, reverse=True)
_UNITES_EN_ML = {"quart de litre": 250, "demi-litre": 500}


def _nombre(s):
    s = s.replace(",", ".").replace(" ", "")
    if s in ("½", "¼", "¾"):
        return {"½": 0.5, "¼": 0.25, "¾": 0.75}[s]
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else None
    return float(s)


def _decouper(ligne):
    """Une ligne peut contenir plusieurs articles : « 1 oignon et 1 gousse d'ail », « Huile d'olive, sel, poivre »."""
    profondeur, morceaux, courant = 0, [], ""
    for c in ligne:
        profondeur += c == "("
        profondeur -= c == ")"
        if c in ",+" and profondeur == 0 and (c == "+" or not re.search(r"\d", ligne)):
            morceaux.append(courant)
            courant = ""
        else:
            courant += c
    morceaux.append(courant)
    out = []
    for m in morceaux:  # « et » : seulement hors parenthèses, entre deux articles chiffrés ou sans aucun chiffre
        parties = re.split(r"\s+et\s+(?![^(]*\))", m)
        if len(parties) > 1 and (all(re.match(rf"\s*{_NOMBRE}", p) for p in parties) or not re.search(r"\d", m)):
            out += parties
        else:
            out.append(m)
    return [o.strip(" .,;:-–•*\t") for o in out if o.strip(" .,;:-–•*\t")]


def lire_article(texte):
    """« 20 cl de vin blanc sec » -> {"nom": "vin blanc sec", "quantite": 20, "unite": "cl"}"""
    t = texte.strip()
    q, unite = None, ""
    fin = re.search(rf"\(\s*({_NOMBRE})\s*([^()]*?)\)\s*$", t)  # « Beurre (100 grammes) »
    if fin and not re.match(rf"^\s*{_NOMBRE}", t):
        q, unite, t = _nombre(fin.group(1)), fin.group(2).strip(), t[:fin.start()].strip()
    else:
        mot = re.match(r"^\s*(\w+)\s+(.*)$", t)
        if mot and norm(mot.group(1)) in _MOTS_NOMBRES:  # « Quatre bananes plantains », « Une pincée de sel »
            t = f"{_MOTS_NOMBRES[norm(mot.group(1))]} {mot.group(2)}"
        m = _QUANTITE.match(t)
        if m and m.group(3):
            q = _nombre(m.group(2) or m.group(1))  # fourchette « 2 à 4 tomates » : on prend le haut
            reste = m.group(3).strip()
            r = norm(reste)
            for u in _UNITES:
                if r == u or r.startswith(u + " ") or r.startswith(u + "."):
                    unite = reste[:len(u)]
                    reste = reste[len(u):].strip(" .")
                    break
            reste = re.sub(r"^(de|d'|d’|des|du)\s*", "", reste, flags=re.I).strip()
            t = reste
    nom = re.sub(r"\s+", " ", t).strip(" .,;:")
    nu = norm(unite)
    if nu in _UNITES_EN_ML:
        q, unite = (q or 1) * _UNITES_EN_ML[nu], "ml"
    if nu in ("grammes", "gramme"):
        unite = "g"
    return {"nom": nom, "quantite": q, "unite": unite} if nom else None


def lire_ingredients(texte):
    """Texte d'une recette -> (nombre de personnes ou None, [articles]). Liste vide si pas de section Ingrédients."""
    lignes = texte.split("\n")
    debut = next((i for i, l in enumerate(lignes) if _DEBUT.match(l)), None)
    if debut is None:
        return None, []
    personnes, articles = None, []
    m = _PERSONNES.search(lignes[debut])
    if m:
        personnes = int(m.group(1))
    for ligne in lignes[debut + 1:]:
        if _FIN.match(ligne):
            break
        brut = ligne.strip()
        if not brut:
            continue
        m = _PERSONNES.search(brut)
        if m and len(brut) < 60:  # « Pour 4 personnes : »
            personnes = personnes or int(m.group(1))
            continue
        if brut.endswith(":") or norm(brut).startswith(("- ingredient", "ingredient")) or len(brut) > 120:
            continue  # sous-titre (« Pâte : », « - Ingrédients pour la sauce ») ou phrase
        for morceau in _decouper(brut):
            a = lire_article(morceau)
            if a and len(a["nom"]) <= 80:
                articles.append(a)
    return personnes, articles
