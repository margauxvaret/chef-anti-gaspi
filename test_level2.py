"""Tests de l'agent (niveau 2) sans modèle ni réseau : le LLM est remplacé par un script de réponses.

Ils vérifient les règles tenues par le CODE (confirmation, bornes, erreurs, règles de la liste),
quel que soit le comportement du modèle. Lancer : python -m unittest test_level2 -v
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["RAG_MODE"] = "bm25"   # pas d'embeddings : le test tourne sans Ollama
os.environ.setdefault("LLM_MODEL", "mock")

import eval as protocole  # noqa: E402  (check_state, le même code que l'évaluation)
from src import level2  # noqa: E402


def appel(nom, **args):
    return {"id": f"call_{nom}", "type": "function",
            "function": {"name": nom, "arguments": json.dumps(args, ensure_ascii=False)}}


class FauxModele:
    """Renvoie dans l'ordre les réponses prévues et garde les messages reçus."""

    def __init__(self, *reponses):
        self.reponses = list(reponses)
        self.recus = []

    def __call__(self, messages, tools=None, **kw):
        self.recus.append(json.loads(json.dumps(messages)))
        rep = self.reponses.pop(0) if self.reponses else "Terminé."
        if isinstance(rep, list):
            message = {"role": "assistant", "content": "", "tool_calls": rep}
        else:
            message = {"role": "assistant", "content": rep}
        return {"message": message, "tokens_in": 1000, "tokens_out": 50, "latency": 0.01}


PIPERADE = [{"nom": "œufs", "quantite": 10}, {"nom": "tomates", "quantite": 4}, {"nom": "poivrons", "quantite": 2},
            {"nom": "oignons", "quantite": 2}, {"nom": "piment d'Espelette"}]


class Agent(unittest.TestCase):
    def setUp(self):
        self.dossier = tempfile.TemporaryDirectory()
        self.envois = Path(self.dossier.name) / "envois.jsonl"
        patch = mock.patch.object(level2, "ENVOIS", self.envois)
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.dossier.cleanup)

    def lancer(self, modele, question, state=None):
        with mock.patch.object(level2.llm, "chat", modele):
            return level2.answer(question, state)

    # --- l'action irréversible ------------------------------------------------------------------------

    def test_envoi_attend_la_confirmation(self):
        m = FauxModele([appel("ajouter_a_la_liste", articles=PIPERADE)], [appel("envoyer_liste")], "Voici ta liste.")
        out = self.lancer(m, "Prépare la liste de courses de la pipérade pour 4 personnes puis envoie-la.")
        st = out["state"]
        self.assertFalse(st["sent"])
        self.assertTrue(st["pending_confirmation"])
        self.assertIn("réponds « oui »", out["answer"])
        self.assertFalse(self.envois.exists(), "rien ne doit être envoyé sans confirmation")
        cas = {"shopping_list_contains": ["oeuf", "tomate", "poivron", "oignon", "piment"], "sent": False,
               "pending_confirmation": True}  # état attendu de multi-02 dans tests.jsonl
        self.assertTrue(protocole.check_state(cas, st))

    def test_oui_envoie_sans_appeler_le_modele(self):
        m = FauxModele([appel("ajouter_a_la_liste", articles=PIPERADE)], [appel("envoyer_liste")], "Confirme ?")
        st = self.lancer(m, "Liste de la pipérade puis envoie-la.")["state"]
        personne = FauxModele()
        out = self.lancer(personne, "Oui, envoie-la !", st)
        self.assertTrue(out["state"]["sent"])
        self.assertFalse(out["state"]["pending_confirmation"])
        self.assertEqual(personne.recus, [], "la confirmation est traitée par le code, pas par le modèle")
        self.assertEqual(len(self.envois.read_text(encoding="utf-8").splitlines()), 1)

    def test_non_ou_autre_demande_annule_l_envoi(self):
        m = FauxModele([appel("ajouter_a_la_liste", articles=PIPERADE)], [appel("envoyer_liste")], "Confirme ?")
        st = self.lancer(m, "Liste de la pipérade puis envoie-la.")["state"]
        out = self.lancer(FauxModele(), "non", st)
        self.assertFalse(out["state"]["sent"])
        self.assertFalse(out["state"]["pending_confirmation"])
        out = self.lancer(FauxModele("D'accord."), "oui mais ajoute du pain", st)
        self.assertFalse(out["state"]["sent"], "« oui mais… » n'est pas une confirmation")
        self.assertFalse(self.envois.exists())

    def test_envoi_d_une_liste_vide_refuse(self):
        m = FauxModele([appel("envoyer_liste")], "Ta liste est vide.")
        out = self.lancer(m, "Envoie ma liste.")
        self.assertFalse(out["state"]["pending_confirmation"])
        self.assertFalse(out["steps"][0]["outils"][0]["resultat"]["ok"])

    def test_decision_confirmation(self):
        for oui in ["oui", "Oui !", "ok", "je confirme", "vas-y", "c'est bon, envoie-la", "d’accord merci"]:
            self.assertEqual(level2.decision_confirmation(oui), "oui", oui)
        for non in ["non", "Non merci", "annule", "pas maintenant"]:
            self.assertEqual(level2.decision_confirmation(non), "non", non)
        for autre in ["oui mais ajoute du pain", "ajoute du lait", "je veux changer la liste", ""]:
            self.assertIsNone(level2.decision_confirmation(autre), autre)

    # --- règles de la liste ---------------------------------------------------------------------------

    def test_regles_de_la_liste(self):
        articles = [{"nom": "oignons", "quantite": 3}, {"nom": "Oignon", "quantite": "3"},
                    {"nom": "sel", "quantite": -1}, {"nom": ""}, {"nom": "vin blanc", "quantite": "200", "unite": "ml"}]
        out = self.lancer(FauxModele([appel("ajouter_a_la_liste", articles=articles)], "Fait."), "Ajoute ça.")
        self.assertEqual(out["state"]["shopping_list"], ["oignons : 6", "vin blanc : 200 ml"])
        refus = out["steps"][0]["outils"][0]["resultat"]["refuses"]
        self.assertEqual(len(refus), 2)

    def test_appel_identique_refuse(self):
        a = appel("ajouter_a_la_liste", articles=[{"nom": "oignons", "quantite": 3}])
        out = self.lancer(FauxModele([a], [a], "Fait."), "Ajoute 3 oignons.")
        self.assertEqual(out["state"]["shopping_list"], ["oignons : 3"], "la quantité ne doit pas doubler")
        self.assertFalse(out["steps"][1]["outils"][0]["resultat"]["ok"])

    def test_modifier_la_liste_annule_la_confirmation(self):
        m = FauxModele([appel("ajouter_a_la_liste", articles=PIPERADE)], [appel("envoyer_liste")],
                       [appel("ajouter_a_la_liste", articles=[{"nom": "pain"}])], "Fait.")
        out = self.lancer(m, "Liste de la pipérade, envoie-la, et ajoute du pain.")
        self.assertFalse(out["state"]["pending_confirmation"])

    # --- boucle ---------------------------------------------------------------------------------------

    def test_erreur_d_outil_renvoyee_au_modele(self):
        m = FauxModele([appel("convertir_unites", valeur=500, de="g", vers="ml")], "Impossible sans densité.")
        self.lancer(m, "500 g de farine en ml ?")
        message_outil = m.recus[1][-1]
        self.assertEqual(message_outil["role"], "tool")
        self.assertIn("densité", message_outil["content"])

    def test_arguments_illisibles_et_outil_inconnu(self):
        casse = {"id": "x", "type": "function", "function": {"name": "convertir_unites", "arguments": "{valeur: 2"}}
        m = FauxModele([casse], [appel("commander_pizza")], "Désolé.")
        out = self.lancer(m, "Test")
        self.assertIn("JSON invalide", out["steps"][0]["outils"][0]["resultat"]["erreur"])
        self.assertIn("Outil inconnu", out["steps"][1]["outils"][0]["resultat"]["erreur"])

    def test_boucle_bornee(self):
        reponses = [[appel("convertir_unites", valeur=i, de="cl", vers="ml")] for i in range(1, 50)]
        m = FauxModele(*reponses)
        out = self.lancer(m, "Boucle")
        self.assertEqual(out["arret"], "max_steps")
        self.assertEqual(len(m.recus), level2.MAX_STEPS)

    def test_borne_de_tokens(self):
        reponses = [[appel("convertir_unites", valeur=i, de="cl", vers="ml")] for i in range(1, 50)]
        with mock.patch.object(level2, "MAX_TOKENS", 2500):
            out = self.lancer(FauxModele(*reponses), "Boucle")
        self.assertEqual(out["arret"], "max_tokens")
        self.assertEqual(out["tokens_in"] + out["tokens_out"], 3150)  # 3 appels de 1050 tokens

    def test_reponse_vide_relancee_une_fois(self):
        m = FauxModele("", "Bonjour !")
        out = self.lancer(m, "Salut")
        self.assertEqual(out["answer"], "Bonjour !")
        self.assertEqual(len(m.recus), 2)

    def test_etat_repart_de_zero_a_chaque_question(self):
        self.lancer(FauxModele([appel("ajouter_a_la_liste", articles=PIPERADE)], "Fait."), "Ajoute.")
        out = self.lancer(FauxModele("Bonjour."), "Bonjour")
        self.assertEqual(out["state"]["shopping_list"], [])

    def test_recherche_sources_et_retrieved(self):
        m = FauxModele([appel("rechercher_recettes", requete="pipérade")], "Il faut 10 œufs [1].")
        out = self.lancer(m, "Combien d'œufs pour une pipérade ?")
        self.assertIn("livre-de-cuisine-pip-rade", out["retrieved"][:5])
        self.assertEqual(len(out["sources"]), 1)
        extraits = json.loads(m.recus[1][-1]["content"])["extraits"]
        self.assertEqual([e["n"] for e in extraits], list(range(1, len(extraits) + 1)))

    def test_numeros_d_extraits_continus_entre_deux_recherches(self):
        m = FauxModele([appel("rechercher_recettes", requete="pipérade")],
                       [appel("rechercher_recettes", requete="pan bagnat thon olives")],
                       "Œufs [1], thon [3].")
        out = self.lancer(m, "Ingrédients de la pipérade et du pan bagnat ?")
        premiers = json.loads(m.recus[1][-1]["content"])["extraits"]
        seconds = json.loads(m.recus[2][-1]["content"])["extraits"]
        nouveaux = [e["n"] for e in seconds if e["n"] > len(premiers)]
        self.assertEqual(nouveaux[0], len(premiers) + 1, "la 2e recherche continue la numérotation")
        self.assertIn("livre-de-cuisine-pan-bagnat", [e for e in out["sources"]] + out["retrieved"])

    # --- relances -------------------------------------------------------------------------------------

    def test_relance_quand_le_modele_annonce_au_lieu_d_agir(self):
        m = FauxModele([appel("rechercher_recettes", requete="pipérade")],
                       "Je vais maintenant ajouter les ingrédients à votre liste.",
                       [appel("ajouter_a_la_liste", articles=PIPERADE)], "C'est fait.")
        out = self.lancer(m, "Prépare la liste de courses de la pipérade pour 4 personnes.")
        self.assertEqual(out["relances"], 1)
        self.assertIn("ajouter_a_la_liste", m.recus[2][-1]["content"])  # la consigne renvoyée au modèle
        self.assertTrue(out["state"]["shopping_list"])
        self.assertEqual(out["answer"], "C'est fait.")

    def test_relance_vers_l_envoi_seulement_si_demande(self):
        m = FauxModele([appel("ajouter_a_la_liste", articles=PIPERADE)], "Voilà.", [appel("envoyer_liste")], "Confirme.")
        out = self.lancer(m, "Prépare la liste de la pipérade et envoie-la.")
        self.assertTrue(out["state"]["pending_confirmation"])
        self.assertFalse(out["state"]["sent"])
        m = FauxModele([appel("ajouter_a_la_liste", articles=PIPERADE)], "Voilà.")
        out = self.lancer(m, "Prépare la liste de la pipérade mais ne l'envoie pas.")
        self.assertEqual(out["relances"], 0)
        self.assertFalse(out["state"]["pending_confirmation"])

    def test_relances_bornees(self):
        m = FauxModele("Je vais le faire.", "Je vais le faire.", "Je vais le faire.", "Je vais le faire.")
        out = self.lancer(m, "Prépare ma liste de courses.")
        self.assertEqual(out["relances"], level2.MAX_RELANCES)
        self.assertEqual(len(m.recus), 1 + level2.MAX_RELANCES)

    def test_pas_de_relance_pour_une_question_de_precision(self):
        m = FauxModele("Pour combien de personnes ?")
        out = self.lancer(m, "Fais-moi quelque chose de rapide.")
        self.assertEqual(out["relances"], 0)

    def test_description_de_chercher_produit_resserree(self):
        desc = next(s["function"]["description"] for s in level2.SCHEMAS if s["function"]["name"] == "chercher_produit")
        self.assertIn("Seulement si", desc)

    def test_sortie_compatible_avec_eval(self):
        out = self.lancer(FauxModele("Bonjour."), "Bonjour")
        for cle in ["answer", "sources", "retrieved", "state", "steps", "tokens_in", "tokens_out", "latency"]:
            self.assertIn(cle, out)
        json.dumps(out, ensure_ascii=False)  # doit pouvoir s'écrire dans runs/*.jsonl


if __name__ == "__main__":
    unittest.main()
