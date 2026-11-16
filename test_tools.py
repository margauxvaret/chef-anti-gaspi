"""Tests des outils (sans réseau). Lancer : python -m unittest test_tools -v"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import tools


class Conversions(unittest.TestCase):
    def test_volume_et_masse(self):
        self.assertEqual(tools.convertir_unites(20, "cl", "ml")["resultat"], 200)
        self.assertEqual(tools.convertir_unites(1.5, "kg", "g")["resultat"], 1500)
        self.assertEqual(tools.convertir_unites("1/2", "l", "ml")["resultat"], 500)
        self.assertEqual(tools.convertir_unites("1,5", "kg", "g")["resultat"], 1500)
        self.assertEqual(tools.convertir_unites(2, "cuillère à soupe", "ml")["resultat"], 30)
        self.assertEqual(tools.convertir_unites(3, "c.à.c.", "ml")["resultat"], 15)
        self.assertEqual(tools.convertir_unites(1, "livre", "g")["resultat"], 500)
        self.assertAlmostEqual(tools.convertir_unites(500, "g", "lb")["resultat"], 1.1023, places=3)

    def test_temperature(self):
        self.assertEqual(tools.convertir_unites(220, "°C", "thermostat")["resultat"], 7)
        self.assertEqual(tools.convertir_unites(6, "thermostat", "°C")["resultat"], 180)
        self.assertEqual(tools.convertir_unites(180, "°C", "°F")["resultat"], 356)
        self.assertEqual(tools.convertir_unites(212, "°F", "°C")["resultat"], 100)

    def test_erreurs_lisibles(self):
        r = tools.convertir_unites(500, "g", "ml")
        self.assertFalse(r["ok"])
        self.assertIn("densité", r["erreur"])
        self.assertFalse(tools.convertir_unites(5, "parsec", "g")["ok"])
        self.assertIn("Unités connues", tools.convertir_unites(5, "parsec", "g")["erreur"])
        self.assertFalse(tools.convertir_unites("beaucoup", "g", "kg")["ok"])
        self.assertFalse(tools.convertir_unites(5, "°C", "g")["ok"])


def fausse_reponse(produits, code=200):
    r = mock.Mock()
    r.status_code = code
    r.json.return_value = {"products": produits, "hits": produits}  # v1 : "products", Search-a-licious : "hits"
    return r


PRODUITS = [
    {"code": "123", "product_name": "Lait de coco", "brands": "Aroy-D", "quantity": "400 ml",
     "nutriscore_grade": "d", "nova_group": 3, "nutriments": {"energy-kcal_100g": 180}},
    {"code": "456", "product_name": "", "brands": "Sans nom"},  # ignoré : pas de nom
]


@mock.patch.object(tools.time, "sleep", lambda s: None)
class RechercheProduit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = mock.patch.object(tools, "CACHE_PATH", Path(self.tmp.name) / "cache.json")
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_succes_puis_cache(self):
        with mock.patch.object(tools.requests, "get", return_value=fausse_reponse(PRODUITS)) as get:
            r1 = tools.chercher_produit("Lait de coco")
            r2 = tools.chercher_produit("lait de  COCO")  # même requête normalisée
        self.assertTrue(r1["ok"])
        self.assertEqual(r1["nombre"], 1)
        self.assertEqual(r1["resultats"][0]["nutriscore"], "D")
        self.assertEqual(r1["resultats"][0]["kcal_100g"], 180)
        self.assertEqual(r1["source"], "open_food_facts")
        self.assertEqual(r2["source"], "cache")
        self.assertEqual(get.call_count, 1)  # une seule requête réseau
        self.assertIn("User-Agent", get.call_args.kwargs["headers"])

    def test_erreurs(self):
        with mock.patch.object(tools.requests, "get", side_effect=tools.requests.ConnectionError("réseau")):
            r = tools.chercher_produit("beurre")
        self.assertFalse(r["ok"])
        self.assertIn("indisponible", r["erreur"])
        with mock.patch.object(tools.requests, "get", return_value=fausse_reponse([], code=429)):
            self.assertIn("10 par minute", tools.chercher_produit("sucre")["erreur"])
        self.assertFalse(tools.chercher_produit("a")["ok"])

    def test_repli_sur_la_recherche_v1(self):
        reponses = [fausse_reponse([], code=503), fausse_reponse(PRODUITS)]
        with mock.patch.object(tools.requests, "get", side_effect=reponses) as get:
            r = tools.chercher_produit("lait de coco")
        self.assertTrue(r["ok"])
        self.assertEqual(r["backend"], "v1")
        self.assertEqual(get.call_count, 2)

    def test_les_deux_recherches_en_panne_donnent_les_codes_http(self):
        with mock.patch.object(tools.requests, "get", side_effect=[fausse_reponse([], 503), fausse_reponse([], 403)]):
            r = tools.chercher_produit("riz")
        self.assertFalse(r["ok"])
        self.assertIn("HTTP 503", r["erreur"])
        self.assertIn("HTTP 403", r["erreur"])

    def test_formats_varies_de_search_a_licious(self):
        bruts = [{"code": "9", "product_name": {"main": "Riz basmati"}, "brands": ["Taureau Ailé", "Marque 2"],
                  "quantity": "1 kg", "nutriscore_grade": "a", "nutriments": {}}]
        with mock.patch.object(tools.requests, "get", return_value=fausse_reponse(bruts)):
            r = tools.chercher_produit("riz basmati")
        self.assertEqual(r["resultats"][0]["nom"], "Riz basmati")
        self.assertEqual(r["resultats"][0]["marque"], "Taureau Ailé, Marque 2")

    def test_aucun_resultat_et_hors_ligne(self):
        with mock.patch.object(tools.requests, "get", return_value=fausse_reponse([])):
            r = tools.chercher_produit("zzzzqq")
        self.assertTrue(r["ok"])
        self.assertEqual(r["nombre"], 0)
        with mock.patch.dict(tools.os.environ, {"OFF_OFFLINE": "1"}):
            with mock.patch.object(tools.requests, "get") as get:
                r = tools.chercher_produit("farine")
            self.assertFalse(r["ok"])
            get.assert_not_called()


class Dispatcher(unittest.TestCase):
    def test_executer(self):
        self.assertEqual(tools.executer("convertir_unites", {"valeur": 20, "de": "cl", "vers": "ml"})["resultat"], 200)
        self.assertIn("Outil inconnu", tools.executer("envoyer_pigeon", {})["erreur"])
        self.assertIn("Arguments invalides", tools.executer("convertir_unites", {"valeur": 1})["erreur"])
        self.assertFalse(tools.executer("convertir_unites", None)["ok"])

    def test_schemas_coherents(self):
        noms = [s["function"]["name"] for s in tools.SCHEMAS]
        self.assertEqual(sorted(noms), sorted(tools.TOOLS))
        for s in tools.SCHEMAS:
            self.assertEqual(s["type"], "function")
            self.assertTrue(set(s["function"]["parameters"]["required"]) <= set(s["function"]["parameters"]["properties"]))


if __name__ == "__main__":
    unittest.main()
