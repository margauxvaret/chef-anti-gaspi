"""Outils de l'agent (niveau 2) : conversion d'unités et recherche Open Food Facts.

Chaque outil est une fonction ordinaire qui renvoie TOUJOURS un dictionnaire :
    {"ok": True, ...}                      en cas de succès
    {"ok": False, "erreur": "message"}     sinon (message écrit pour être relu par le LLM)
Un outil ne lève jamais d'exception : l'agent renvoie simplement le dictionnaire au modèle,
qui peut alors corriger sa demande (exigence « erreurs d'outils renvoyées au modèle »).

Pour brancher ces outils dans l'agent :
    from src.tools import SCHEMAS, executer
    chat(messages, tools=SCHEMAS)                    # schémas au format OpenAI
    resultat = executer(nom_outil, arguments_dict)   # arguments = dict déjà décodé du JSON

Open Food Facts (recherche par nom) :
  - la recherche texte passe par l'API v1 (/cgi/search.pl) : l'API v2 ne fait pas de recherche plein texte ;
  - limite officielle : 10 requêtes de recherche par minute et par IP, donc le module espace ses appels
    (>= 6,5 s) et met les réponses en cache dans data/off_cache.json ;
  - OFF demande un User-Agent personnalisé de la forme NomApp/Version (contact).
  - OFF_OFFLINE=1 : n'utilise QUE le cache (évaluation reproductible, sans réseau). Après un premier
    passage en ligne, committez data/off_cache.json pour que le tableau de résultats soit reproductible.
"""
import json
import os
import time
import unicodedata
from fractions import Fraction
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "off_cache.json"
OFF_URL = "https://world.openfoodfacts.org/cgi/search.pl"  # recherche texte « historique » (API v1)
SAL_URL = "https://search.openfoodfacts.org/search"        # Search-a-licious, le moteur de recherche plein texte d'OFF
CHAMPS = "code,product_name,product_name_fr,brands,quantity,nutriscore_grade,nova_group,nutriments"
USER_AGENT = os.getenv("OFF_USER_AGENT", "ChefAntiGaspi/0.1 (https://github.com/margauxvaret/chef-anti-gaspi)")
MIN_INTERVAL = 6.5  # secondes entre deux recherches en ligne (limite OFF : 10 par minute)
_derniere_requete = 0.0


# ----------------------------------------------------------------------------------------------
# 1. Conversion d'unités
# ----------------------------------------------------------------------------------------------

def _norm(s):
    """Minuscules, sans accents, sans points ni espaces multiples : « C.à.S. » -> « cas »."""
    s = unicodedata.normalize("NFD", str(s).lower().replace("’", "'"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.replace(".", "").split())


UNITES = {}  # nom normalisé -> (dimension, facteur vers l'unité de base : g pour la masse, ml pour le volume)


def _ajouter(dimension, facteur, *noms):
    for n in noms:
        UNITES[_norm(n)] = (dimension, facteur)


_ajouter("masse", 0.001, "mg", "milligramme", "milligrammes")
_ajouter("masse", 1, "g", "gr", "gramme", "grammes")
_ajouter("masse", 1000, "kg", "kilo", "kilos", "kilogramme", "kilogrammes")
_ajouter("masse", 500, "livre", "livres")  # livre « de cuisine » française ; pour la livre anglaise : lb
_ajouter("masse", 453.592, "lb", "lbs", "pound", "pounds")
_ajouter("masse", 28.3495, "oz", "once", "onces", "ounce", "ounces")
_ajouter("volume", 1, "ml", "millilitre", "millilitres")
_ajouter("volume", 10, "cl", "centilitre", "centilitres")
_ajouter("volume", 100, "dl", "decilitre", "decilitres")
_ajouter("volume", 1000, "l", "litre", "litres")
_ajouter("volume", 15, "cas", "cs", "cuillere a soupe", "cuilleres a soupe", "tbsp")
_ajouter("volume", 5, "cac", "cc", "cuillere a cafe", "cuilleres a cafe", "tsp")
_ajouter("volume", 250, "tasse", "tasses")  # approximation
_ajouter("volume", 200, "verre", "verres")  # approximation
_ajouter("volume", 236.588, "cup", "cups")
_ajouter("volume", 29.5735, "fl oz")

TEMPERATURES = {
    "c": "c", "°c": "c", "celsius": "c", "degre": "c", "degres": "c", "degre celsius": "c", "degres celsius": "c",
    "f": "f", "°f": "f", "fahrenheit": "f", "degre fahrenheit": "f", "degres fahrenheit": "f",
    "thermostat": "th", "th": "th",
}


def _parse_valeur(v):
    """Accepte 1.5, '1,5', '1/2', '1 1/2', '½'."""
    if isinstance(v, bool):
        raise ValueError("booléen")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace(",", ".")
    s = s.replace("½", " 1/2").replace("¼", " 1/4").replace("¾", " 3/4")
    parts = s.split()
    if not parts:
        raise ValueError("vide")
    return float(sum(Fraction(p) for p in parts))


def _fmt(x):
    return f"{x:g}"


def convertir_unites(valeur, de, vers):
    """Convertit une quantité de cuisine (masse, volume, température, thermostat).

    Exemples : (20, "cl", "ml") -> 200 ; (2, "cuillère à soupe", "ml") -> 30 ; (220, "°C", "thermostat") -> 7.
    Ne convertit PAS une masse en volume (il faudrait la densité de l'ingrédient).
    """
    try:
        x = _parse_valeur(valeur)
    except (ValueError, ZeroDivisionError):
        return {"ok": False, "erreur": f"Valeur non numérique : {valeur!r}. Donner un nombre, ex. 20 ou 1,5 ou 1/2."}

    a, b = _norm(de), _norm(vers)

    # température et thermostat
    if a in TEMPERATURES or b in TEMPERATURES:
        if not (a in TEMPERATURES and b in TEMPERATURES):
            return {"ok": False, "erreur": f"Impossible de convertir « {de} » en « {vers} » : une température ne se convertit qu'en °C, °F ou thermostat."}
        ua, ub = TEMPERATURES[a], TEMPERATURES[b]
        celsius = {"c": x, "f": (x - 32) * 5 / 9, "th": x * 30}[ua]
        res = {"c": celsius, "f": celsius * 9 / 5 + 32, "th": round(celsius / 30)}[ub]
        res = round(res, 2)
        out = {"ok": True, "valeur": x, "de": de, "vers": vers, "resultat": res,
               "texte": f"{_fmt(x)} {de} = {_fmt(res)} {vers}"}
        if "th" in (ua, ub):
            out["remarque"] = "Thermostat approximatif : th = °C / 30 (ex. th 6 = 180 °C, th 7 = 210 °C)."
        return out

    for nom, orig in ((a, de), (b, vers)):
        if nom not in UNITES:
            connues = "g, kg, mg, lb, oz, ml, cl, dl, l, cuillère à soupe, cuillère à café, tasse, verre, cup, °C, °F, thermostat"
            return {"ok": False, "erreur": f"Unité inconnue : {orig!r}. Unités connues : {connues}."}
    (dim_a, fa), (dim_b, fb) = UNITES[a], UNITES[b]
    if dim_a != dim_b:
        return {"ok": False, "erreur": f"Impossible de convertir une {dim_a} ({de}) en {dim_b} ({vers}) sans connaître la densité de l'ingrédient."}
    res = round(x * fa / fb, 4)
    out = {"ok": True, "valeur": x, "de": de, "vers": vers, "resultat": res,
           "texte": f"{_fmt(x)} {de} = {_fmt(res)} {vers}"}
    if a in ("tasse", "tasses", "verre", "verres") or b in ("tasse", "tasses", "verre", "verres"):
        out["remarque"] = "Tasse (250 ml) et verre (200 ml) sont des valeurs approximatives."
    if a in ("livre", "livres") or b in ("livre", "livres"):
        out["remarque"] = "Livre de cuisine française = 500 g (la livre anglaise, lb, vaut 453,6 g)."
    return out


# ----------------------------------------------------------------------------------------------
# 2. Recherche Open Food Facts
# ----------------------------------------------------------------------------------------------

def _charger_cache():
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _sauver_cache(cache):
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # le cache est un confort, pas une obligation


def _attendre_quota():
    global _derniere_requete
    delta = time.time() - _derniere_requete
    if delta < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - delta)
    _derniere_requete = time.time()


class _Limite(Exception):
    """Open Food Facts a répondu 429 (trop de requêtes)."""


def _get_json(url, params):
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    if r.status_code == 429:
        raise _Limite()
    if r.status_code >= 400:
        raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
    return r.json()


def _produits_sal(nom, n):
    return _get_json(SAL_URL, {"q": nom, "page_size": n, "langs": "fr", "fields": CHAMPS}).get("hits", [])


def _produits_v1(nom, n):
    data = _get_json(OFF_URL, {"search_terms": nom, "search_simple": 1, "action": "process", "json": 1,
                               "page_size": n, "cc": "fr", "lc": "fr", "fields": CHAMPS})
    return data.get("products", [])


def _texte(v):
    """Les champs d'OFF sont tantôt du texte, tantôt une liste ou un dict par langue : on aplatit."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("fr", "main", "en"):
            if v.get(k):
                return _texte(v[k])
        return next((_texte(x) for x in v.values() if x), "")
    if isinstance(v, (list, tuple)):
        return ", ".join(_texte(x) for x in v if x)
    return str(v)


def _resume(p):
    nutriments = p.get("nutriments") or {}
    code = p.get("code")
    return {
        "nom": _texte(p.get("product_name_fr")) or _texte(p.get("product_name")),
        "marque": _texte(p.get("brands")),
        "quantite": _texte(p.get("quantity")),
        "nutriscore": _texte(p.get("nutriscore_grade")).upper() or None,
        "nova": p.get("nova_group"),
        "kcal_100g": nutriments.get("energy-kcal_100g"),
        "code": code,
        "url": f"https://fr.openfoodfacts.org/produit/{code}" if code else None,
    }


def chercher_produit(nom, max_resultats=3):
    """Cherche un produit alimentaire par son nom dans Open Food Facts.

    Retourne {"ok": True, "recherche", "nombre", "resultats": [{"nom", "marque", "quantite",
    "nutriscore", "nova", "kcal_100g", "code", "url"}], "source": "open_food_facts" | "cache"}.
    """
    if not isinstance(nom, str) or len(nom.strip()) < 2:
        return {"ok": False, "erreur": "Nom de produit trop court : donner au moins 2 caractères, ex. « lait de coco »."}
    try:
        n = max(1, min(5, int(max_resultats)))
    except (TypeError, ValueError):
        n = 3

    cle = _norm(nom)
    cache = _charger_cache()
    if cle in cache:
        res = dict(cache[cle])
        res["resultats"] = res["resultats"][:n]
        res["nombre"] = len(res["resultats"])
        return {**res, "source": "cache"}
    if os.getenv("OFF_OFFLINE") == "1":
        return {"ok": False, "erreur": f"Mode hors ligne (OFF_OFFLINE=1) et « {nom} » n'est pas dans le cache."}

    # on essaie Search-a-licious, puis la recherche historique v1 si la première échoue ou ne trouve rien
    erreurs, produits, backend = [], [], None
    for nom_backend, fn in (("search-a-licious", _produits_sal), ("v1", _produits_v1)):
        try:
            _attendre_quota()
            bruts = fn(nom, n)
        except _Limite:
            return {"ok": False, "erreur": "Open Food Facts limite le nombre de recherches (10 par minute) : réessayer dans une minute."}
        except requests.HTTPError as e:
            erreurs.append(f"{nom_backend} : {e}")
            continue
        except (requests.RequestException, ValueError) as e:
            erreurs.append(f"{nom_backend} : {type(e).__name__}")
            continue
        produits = [p for p in (_resume(x) for x in bruts) if p["nom"]][:n]
        backend = nom_backend
        if produits:
            break
    if backend is None:
        return {"ok": False, "erreur": "Open Food Facts est indisponible (" + " ; ".join(erreurs)
                + ") : réessayer plus tard ou continuer sans ce produit."}

    res = {"ok": True, "recherche": nom, "nombre": len(produits), "resultats": produits}
    if not produits:
        return {**res, "remarque": f"Aucun produit trouvé pour « {nom} » : essayer un nom plus simple.",
                "source": "open_food_facts", "backend": backend}
    cache[cle] = res
    _sauver_cache(cache)
    return {**res, "source": "open_food_facts", "backend": backend}


# ----------------------------------------------------------------------------------------------
# 3. Branchement dans l'agent
# ----------------------------------------------------------------------------------------------

TOOLS = {"convertir_unites": convertir_unites, "chercher_produit": chercher_produit}

SCHEMAS = [
    {"type": "function", "function": {
        "name": "convertir_unites",
        "description": "Convertit une quantité de cuisine d'une unité à une autre : masse (g, kg, lb, oz), volume (ml, cl, l, "
                       "cuillère à soupe, tasse...), température (°C, °F, thermostat). Ne convertit pas une masse en volume.",
        "parameters": {"type": "object", "properties": {
            "valeur": {"type": "number", "description": "Quantité à convertir, ex. 20"},
            "de": {"type": "string", "description": "Unité de départ, ex. cl, g, cuillère à soupe, °C"},
            "vers": {"type": "string", "description": "Unité d'arrivée, ex. ml, kg, thermostat"},
        }, "required": ["valeur", "de", "vers"]},
    }},
    {"type": "function", "function": {
        "name": "chercher_produit",
        "description": "Cherche un produit alimentaire du commerce par son nom dans Open Food Facts (marque, quantité, "
                       "Nutri-Score, calories pour 100 g). Utile pour choisir un produit à ajouter à la liste de courses.",
        "parameters": {"type": "object", "properties": {
            "nom": {"type": "string", "description": "Nom du produit, ex. lait de coco"},
            "max_resultats": {"type": "integer", "description": "Nombre de résultats (1 à 5, défaut 3)"},
        }, "required": ["nom"]},
    }},
]


def executer(nom, arguments):
    """Exécute l'outil `nom` avec le dict `arguments`. Ne lève jamais d'exception."""
    fn = TOOLS.get(nom)
    if fn is None:
        return {"ok": False, "erreur": f"Outil inconnu : {nom!r}. Outils disponibles : {', '.join(TOOLS)}."}
    try:
        return fn(**(arguments or {}))
    except TypeError as e:
        return {"ok": False, "erreur": f"Arguments invalides pour {nom} : {e}"}
    except Exception as e:  # filet de sécurité : l'agent ne doit jamais planter à cause d'un outil
        return {"ok": False, "erreur": f"Erreur dans l'outil {nom} : {type(e).__name__}: {e}"}


if __name__ == "__main__":
    print(convertir_unites(20, "cl", "ml"))
    print(convertir_unites(220, "°C", "thermostat"))
    print(convertir_unites(500, "g", "ml"))
    print(chercher_produit("lait de coco"))
