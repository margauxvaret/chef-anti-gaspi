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
from src import recettes as R  # noqa: E402


def appel(outil_, **args):
    return {"id": f"call_{outil_}", "type": "function",
            "function": {"name": outil_, "arguments": json.dumps(args, ensure_ascii=False)}}


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


class Recettes(unittest.TestCase):
    def test_lire_les_formes_courantes(self):
        cas = {"10 œufs": ("œufs", 10, ""), "20 cl de vin blanc sec,": ("vin blanc sec", 20, "cl"),
               "2 gousses d'ail": ("ail", 2, "gousses"), "Beurre (100 grammes)": ("Beurre", 100, "g"),
               "Quatre bananes plantains": ("bananes plantains", 4, ""), "2 à 4 tomates": ("tomates", 4, ""),
               "16 cuillères à soupe huile d'olive vierge": ("huile d'olive vierge", 16, "cuillères à soupe"),
               "sel": ("sel", None, "")}
        for texte, (nom, q, unite) in cas.items():
            a = R.lire_article(texte)
            self.assertEqual((a["nom"], a["quantite"], a["unite"]), (nom, q, unite), texte)

    def test_lignes_a_plusieurs_articles(self):
        _, articles = R.lire_ingredients("Ingrédients\nPour 4 personnes :\n1 oignon et 1 gousse d'ail\n"
                                         "Huile d'olive, sel, poivre\n100 g de poulet (ou crevette, ou tofu)\n"
                                         "Préparation\nCouper.")
        self.assertEqual([a["nom"] for a in articles],
                         ["oignon", "ail", "Huile d'olive", "sel", "poivre", "poulet (ou crevette, ou tofu)"])

    def test_recettes_des_tests(self):
        for nom, personnes, attendus in [("pipérade", 4, ["œufs", "poivron rouge"]), ("pad thaï", 2, ["cacahuètes"]),
                                         ("hachis parmentier", 4, ["Beurre", "Lait"]), ("escalivada", None, ["aubergines"])]:
            doc = R.trouver_recette(nom)
            p, articles = R.lire_ingredients(R.recettes()[doc]["texte"])
            self.assertEqual(p, personnes, nom)
            noms = " | ".join(a["nom"] for a in articles)
            for x in attendus:
                self.assertIn(x, noms, nom)
        self.assertIsNone(R.trouver_recette("tiramisu"))

    def test_mots_entiers(self):
        self.assertTrue(R.contient("gousse d'ail", "ail"))
        self.assertFalse(R.contient("bouillon de volaille", "ail"))
        self.assertTrue(R.contient("pomme de terre « farineuses »", "pommes de terre"))


class Routage(unittest.TestCase):
    def test_questions_vers_le_rag_actions_vers_l_agent(self):
        rag = ["Combien de temps faut-il blanchir les feuilles de vigne avant de les farcir ?",
               "Pour combien de personnes est prévue la recette de bisque de homard ?",
               "Qu'est-ce que je peux faire avec ça ?", "Quel est le cours du Bitcoin aujourd'hui ?",
               "Quel légume de la famille des choux est ajouté dans le goulash ?",
               "Combien de temps cuit la viande avant d'ajouter les carottes ?"]
        rag += [c["question"] for c in protocole.load_tests() if c["type"] != "multi_step"]
        agent = ["Convertis 20 cl en millilitres.", "250 grammes, ça fait combien en onces ?",
                 "Quel est le Nutri-Score du yaourt nature ?", "Ajoute du pain.", "Envoie-la."]
        agent += [c["question"] for c in protocole.load_tests() if c["type"] == "multi_step"]
        for q in rag:
            self.assertEqual(level2.route(q), "rag", q)
        for q in agent:
            self.assertEqual(level2.route(q), "agent", q)

    def test_une_question_passe_par_le_rag_du_niveau_1(self):
        reponse = {"answer": "Il faut les blanchir 35 secondes [1].", "sources": ["livre-de-cuisine-x"],
                   "retrieved": ["livre-de-cuisine-x"], "state": {}, "steps": [{"mode": "bm25"}],
                   "tokens_in": 300, "tokens_out": 20, "latency": 0.1}
        modele = FauxModele()
        with mock.patch.object(level2, "ROUTAGE", True), mock.patch.object(level2.level1, "answer", return_value=reponse), \
                mock.patch.object(level2.llm, "chat", modele):
            out = level2.answer("Combien de temps faut-il blanchir les feuilles de vigne ?")
        self.assertEqual(out["route"], "rag")
        self.assertEqual(out["answer"], reponse["answer"])
        self.assertEqual(out["sources"], ["livre-de-cuisine-x"])
        self.assertEqual(out["tokens_in"], 300)
        self.assertEqual(modele.recus, [], "l'agent n'est pas appelé")


class Agent(unittest.TestCase):
    """La boucle de l'agent, routage coupé (les questions de test simples iraient sinon au RAG)."""

    def setUp(self):
        patch_routage = mock.patch.object(level2, "ROUTAGE", False)
        patch_routage.start()
        self.addCleanup(patch_routage.stop)
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
        m = FauxModele([appel("ajouter_recette", recette="pipérade")], [appel("envoyer_liste")], "Voici ta liste.")
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
        m = FauxModele([appel("ajouter_recette", recette="pipérade")], [appel("envoyer_liste")], "Confirme ?")
        st = self.lancer(m, "Liste de la pipérade puis envoie-la.")["state"]
        personne = FauxModele()
        out = self.lancer(personne, "Oui, envoie-la !", st)
        self.assertTrue(out["state"]["sent"])
        self.assertFalse(out["state"]["pending_confirmation"])
        self.assertEqual(personne.recus, [], "la confirmation est traitée par le code, pas par le modèle")
        self.assertEqual(len(self.envois.read_text(encoding="utf-8").splitlines()), 1)

    def test_non_ou_autre_demande_annule_l_envoi(self):
        m = FauxModele([appel("ajouter_recette", recette="pipérade")], [appel("envoyer_liste")], "Confirme ?")
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
        m = FauxModele([appel("ajouter_recette", recette="pipérade")], [appel("envoyer_liste")],
                       [appel("ajouter_a_la_liste", articles=[{"nom": "pain"}])], "Fait.")
        out = self.lancer(m, "Liste de la pipérade, envoie-la, et ajoute du pain.")
        self.assertFalse(out["state"]["pending_confirmation"])

    def test_regle_de_trois_faite_par_le_code(self):
        articles = [{"nom": "oignons", "quantite": 3}, {"nom": "riz", "quantite": 150, "unite": "g"},
                    {"nom": "poivron", "quantite": 1}]
        m = FauxModele([appel("ajouter_a_la_liste", articles=articles, personnes_recette=4, personnes=8)], "Fait.")
        out = self.lancer(m, "Pour 8 personnes au lieu de 4 : ajoute ces articles à ma liste.")
        self.assertEqual(out["state"]["shopping_list"], ["oignons : 6", "riz : 300 g", "poivron : 2"])
        self.assertIn("regle_de_trois", out["steps"][0]["outils"][0]["resultat"])
        self.assertTrue(protocole.check_state({"shopping_list_contains": ["oignon", "6"]}, out["state"]))  # multi-08
        m = FauxModele([appel("ajouter_a_la_liste", articles=[{"nom": "œufs", "quantite": 10},
                                                              {"nom": "poivron rouge", "quantite": 1}],
                              personnes_recette=4, personnes=2)], "Fait.")
        out = self.lancer(m, "Liste pour 2 personnes au lieu de 4.")
        self.assertEqual(out["state"]["shopping_list"], ["œufs : 5", "poivron rouge : 1"], "on achète des pièces entières")

    def test_unites_ramenees_en_g_et_ml_puis_regroupees(self):
        m = FauxModele([appel("ajouter_a_la_liste", articles=[
            {"nom": "vin blanc", "quantite": 20, "unite": "cl"}, {"nom": "vin blanc", "quantite": 100, "unite": "ml"},
            {"nom": "œufs", "quantite": 4, "unite": "unité"}, {"nom": "paleron", "quantite": 1, "unite": "kg"}])], "Fait.")
        out = self.lancer(m, "Ajoute ça.")
        self.assertEqual(out["state"]["shopping_list"], ["vin blanc : 300 ml", "œufs : 4", "paleron : 1000 g"])
        m = FauxModele([appel("ajouter_a_la_liste", articles=[{"nom": "vin blanc", "quantite": 100, "unite": "ml"}])],
                       [appel("ajouter_recette", recette="Chili con carne", seulement=["vin blanc"])], "Ok.")
        out = self.lancer(m, "Ajoute le vin blanc du chili con carne en millilitres à ma liste.")
        self.assertIn("ajouter_recette", out["steps"][0]["outils"][0]["resultat"]["erreur"])  # quantité inventée refusée
        self.assertTrue(protocole.check_state({"shopping_list_contains": ["vin blanc", "200"]}, out["state"]))  # multi-04

    def test_allergie_declaree_refusee_et_gardee(self):
        m = FauxModele([appel("ajouter_recette", recette="pad thaï", personnes=2)], "Fait.")
        out = self.lancer(m, "Je suis allergique aux arachides. Prépare la liste de courses du pad thaï pour 2 personnes.")
        st = out["state"]
        self.assertEqual(st["allergies"], ["arachide"])
        self.assertIn("arachide", m.recus[0][0]["content"], "les allergies sont rappelées au modèle")
        self.assertIn("allergie", out["steps"][0]["outils"][0]["resultat"]["refuses"][0]["raison"])
        cas = {"shopping_list_contains": ["riz", "oeuf", "citron"], "shopping_list_excludes": ["cacahuete", "arachide"]}
        self.assertTrue(protocole.check_state(cas, st))  # multi-07
        m = FauxModele([appel("ajouter_a_la_liste", articles=[{"nom": "beurre de cacahuète"}, {"nom": "laitue"}])], "Ok.")
        out = self.lancer(m, "Ajoute du beurre de cacahuète et une laitue.", st)
        self.assertEqual(out["state"]["shopping_list"][-1], "laitue", "l'allergie reste d'un tour à l'autre")

    def test_allergies_declarees(self):
        self.assertEqual(level2.allergies_declarees("allergique au lait et aux œufs, prépare la liste"), ["lait", "oeuf"])
        self.assertEqual(level2.allergies_declarees("J'ai une allergie aux fruits à coque !"), ["fruits a coque"])
        self.assertEqual(level2.allergies_declarees("Prépare la liste du pad thaï"), [])

    # --- recherche ------------------------------------------------------------------------------------

    def test_liste_une_recette_complete(self):
        m = FauxModele([appel("rechercher_recettes", requete="pipérade")], "Fait.")
        self.lancer(m, "Prépare la liste de courses de la pipérade pour 4 personnes.")
        res = json.loads(m.recus[1][-1]["content"])
        self.assertEqual(res["recette"], "Pipérade")
        self.assertEqual(res["personnes_recette"], 4)
        self.assertNotIn("attention", res)
        docs = {level2.level1.get_index().chunks[i]["doc_id"] for i in range(len(level2.level1.get_index().chunks))
                if level2.level1.get_index().chunks[i]["text"] in [e["texte"] for e in res["extraits"]]}
        self.assertEqual(docs, {"livre-de-cuisine-pip-rade"}, "une seule recette, pas de mélange")

    def test_attention_si_le_plat_demande_n_est_pas_dans_le_livre(self):
        chunks = level2.level1.get_index().chunks
        autre_recette = [i for i, c in enumerate(chunks) if c["doc_id"] == "livre-de-cuisine-pip-rade"]
        m = FauxModele([appel("rechercher_recettes", requete="tiramisu")], "Fait.")
        with mock.patch.object(level2.level1, "retrieve", return_value=(autre_recette, None)):
            self.lancer(m, "Prépare la liste de courses du tiramisu pour 4 personnes.")
        res = json.loads(m.recus[1][-1]["content"])
        self.assertIn("tiramisu", res["attention"])
        self.assertFalse(level2.titre_correspond("tiramisu", "Pizza au tofu et aux champignons"))
        self.assertTrue(level2.titre_correspond("chili con carne pour 4 personnes", "Chili con carne"))

    # --- boucle ---------------------------------------------------------------------------------------

    def test_relance_vers_l_envoi_meme_apres_deux_relances_de_liste(self):
        m = FauxModele([appel("rechercher_recettes", requete="pipérade")], "Voici la recette.", "Je vais faire la liste.",
                       [appel("ajouter_recette", recette="pipérade", personnes=4)],
                       "Ta liste est prête.", [appel("envoyer_liste")], "Confirme l'envoi.")
        out = self.lancer(m, "Prépare la liste de courses de la pipérade pour 4 personnes puis envoie-la.")
        self.assertEqual(out["relances_par_motif"], {"liste_vide": 2, "envoi": 1})
        self.assertTrue(out["state"]["pending_confirmation"])  # multi-02

    def test_jamais_de_reponse_vide(self):
        m = FauxModele([appel("ajouter_a_la_liste", articles=[{"nom": "lait de coco", "quantite": 1}])], "", "")
        out = self.lancer(m, "Ajoute un lait de coco.")
        self.assertTrue(out["answer"].startswith("Ta liste de courses"))


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
        self.lancer(FauxModele([appel("ajouter_recette", recette="pipérade")], "Fait."), "Ajoute.")
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
                       [appel("ajouter_recette", recette="pipérade")], "C'est fait.")
        out = self.lancer(m, "Prépare la liste de courses de la pipérade pour 4 personnes.")
        self.assertEqual(out["relances"], 1)
        self.assertIn("ajouter_a_la_liste", m.recus[2][-1]["content"])  # la consigne renvoyée au modèle
        self.assertTrue(out["state"]["shopping_list"])
        self.assertEqual(out["answer"], "C'est fait.")

    def test_relance_vers_l_envoi_seulement_si_demande(self):
        m = FauxModele([appel("ajouter_recette", recette="pipérade")], "Voilà.", [appel("envoyer_liste")], "Confirme.")
        out = self.lancer(m, "Prépare la liste de la pipérade et envoie-la.")
        self.assertTrue(out["state"]["pending_confirmation"])
        self.assertFalse(out["state"]["sent"])
        m = FauxModele([appel("ajouter_recette", recette="pipérade")], "Voilà.")
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

    # --- ajouter_recette : le code lit les ingrédients --------------------------------------------------

    def test_si_le_modele_choisit_bien_les_10_taches_multi_etapes_reussissent(self):
        """Pour chaque cas multi_step de tests.jsonl : les appels qu'un bon agent ferait -> état final vérifié par
        le même code que eval.py. Sépare les erreurs du code (ce test) de celles du modèle (l'évaluation)."""
        bons_appels = {
            "multi-01": [[appel("ajouter_recette", recette="hachis parmentier", personnes=4,
                                sauf=["pommes de terre", "oignons", "ail"])]],
            "multi-02": [[appel("ajouter_recette", recette="pipérade", personnes=4)], [appel("envoyer_liste")]],
            "multi-03": [[appel("ajouter_recette", recette="chili con carne", personnes=4,
                                sauf=["la viande", "le vin blanc", "les épices"])]],
            "multi-04": [[appel("ajouter_recette", recette="chili con carne", seulement=["vin blanc"])]],
            "multi-05": [[appel("chercher_produit", nom="lait de coco en boîte")],
                         [appel("ajouter_a_la_liste", articles=[{"nom": "lait de coco en boîte", "quantite": 1}])]],
            "multi-06": [[appel("ajouter_recette", recette="pipérade"), appel("ajouter_recette", recette="pan bagnat")]],
            "multi-07": [[appel("ajouter_recette", recette="pad thaï", personnes=2)]],
            "multi-08": [[appel("ajouter_recette", recette="chili con carne", personnes=8, seulement=["oignons"])]],
            "multi-09": [[appel("ajouter_recette", recette="escalivada", personnes=4)], [appel("envoyer_liste")]],
            "multi-10": [[appel("ajouter_recette", recette="tiramisu", personnes=4)],
                         [appel("ajouter_recette", recette="pipérade", personnes=2)]],
        }
        cas = [c for c in protocole.load_tests() if c["type"] == "multi_step"]
        self.assertEqual({c["id"] for c in cas}, set(bons_appels))
        reel = level2.executer_tool
        off = lambda nom, args: {"ok": True, "produits": []} if nom == "chercher_produit" else reel(nom, args)  # noqa: E731
        for c in cas:  # routage activé : ces demandes doivent aller à l'agent
            with self.subTest(cas=c["id"]), mock.patch.object(level2, "executer_tool", off), \
                    mock.patch.object(level2, "ROUTAGE", True):
                out = self.lancer(FauxModele(*bons_appels[c["id"]], "Fait."), c["question"])
                self.assertTrue(protocole.check_state(c["final_state"], out["state"]), out["state"]["shopping_list"])

    def test_ajouter_recette_inconnue_ou_ingredient_absent(self):
        m = FauxModele([appel("ajouter_recette", recette="tiramisu")],
                       [appel("ajouter_recette", recette="pipérade", seulement=["mascarpone"])], "Désolé.")
        out = self.lancer(m, "Liste du tiramisu, sinon le mascarpone de la pipérade")
        self.assertIn("Pas de recette « tiramisu »", out["steps"][0]["outils"][0]["resultat"]["erreur"])
        self.assertIn("Aucun ingrédient", out["steps"][1]["outils"][0]["resultat"]["erreur"])
        self.assertEqual(out["state"]["shopping_list"], [])

    def test_sauf_comprend_les_categories(self):
        m = FauxModele([appel("ajouter_recette", recette="chili con carne", sauf="la viande, les épices et le vin blanc")],
                       "Fait.")
        out = self.lancer(m, "Liste du chili con carne, j'ai déjà la viande, les épices et le vin blanc.")
        res = out["steps"][0]["outils"][0]["resultat"]
        self.assertIn("cumin", " ".join(res["retires"]))
        self.assertIn("paleron ou de basse côte", res["retires"])
        self.assertTrue(any("haricots" in x for x in out["state"]["shopping_list"]))

    def test_sauf_et_seulement_non_mentionnes_sont_ignores(self):
        """Run du 17/11 (multi-02) : le modèle retirait les poivrons sans que l'utilisateur l'ait demandé."""
        m = FauxModele([appel("ajouter_recette", recette="Pipérade", personnes=4, sauf=["poivron rouge", "poivron vert"])],
                       "Fait.")
        out = self.lancer(m, "Prépare la liste de courses de la pipérade pour 4 personnes.")
        res = out["steps"][0]["outils"][0]["resultat"]
        self.assertEqual(res["ignores"]["sauf"], ["poivron rouge", "poivron vert"])
        self.assertTrue(any("poivron" in x for x in out["state"]["shopping_list"]))

    def test_recette_non_demandee_refusee(self):
        """Run du 17/11 (multi-10) : la pizza proposée par la recherche au lieu de la pipérade demandée en repli."""
        m = FauxModele([appel("ajouter_recette", recette="Pizza au tofu et aux champignons2", personnes=4)],
                       [appel("ajouter_recette", recette="pipérade", personnes=2)], "Fait.")
        out = self.lancer(m, "Prépare la liste de courses du tiramisu pour 4 personnes. Si cette recette n'existe pas "
                             "dans ton livre de cuisine, prends plutôt la pipérade pour 2 personnes.")
        self.assertIn("Pipérade", out["steps"][0]["outils"][0]["resultat"]["erreur"])
        self.assertIn("œufs : 5", out["state"]["shopping_list"])

    def test_recette_trouvee_par_la_recherche_acceptee_si_aucune_n_est_citee(self):
        question = "Trouve-moi un plat avec mes restes et mets ses ingrédients dans ma liste."
        self.assertEqual(level2.R.recettes_citees([question]), [])
        out = self.lancer(FauxModele([appel("ajouter_recette", recette="escalivada")], "Fait."), question)
        self.assertTrue(out["state"]["shopping_list"])

    def test_ingredient_d_une_recette_citee_pris_dans_le_livre(self):
        """Run du 17/11 (multi-08) : « oignons : 3 cuillère à soupe » et « oignons : 2 g » inventés."""
        m = FauxModele([appel("ajouter_a_la_liste", articles=[{"nom": "oignons", "quantite": 1.5, "unite": "cuillère à soupe"}])],
                       [appel("ajouter_recette", recette="chili con carne", personnes=8, seulement=["oignons"])], "Fait.")
        out = self.lancer(m, "Je cuisine le chili con carne pour 8 personnes au lieu de 4. Ajoute à ma liste les oignons "
                             "avec la bonne quantité.")
        self.assertEqual(out["state"]["shopping_list"], ["oignons : 6"])

    def test_produit_cherche_dans_open_food_facts_pas_redirige(self):
        """multi-05 : un produit trouvé dans Open Food Facts peut être ajouté tel quel, même s'il est dans la recette."""
        reel = level2.executer_tool
        faux = lambda nom, args: {"ok": True, "produits": [{"nom": "Lait de coco"}]} if nom == "chercher_produit" else reel(nom, args)  # noqa: E731
        m = FauxModele([appel("chercher_produit", nom="lait de coco en boîte")],
                       [appel("ajouter_a_la_liste", articles=[{"nom": "lait de coco en boîte", "quantite": 1}])], "Fait.")
        with mock.patch.object(level2, "executer_tool", faux):
            out = self.lancer(m, "Cherche dans Open Food Facts un lait de coco en boîte et ajoute-le à ma liste de courses "
                                 "pour le porc au lait de coco.")
        self.assertEqual(out["state"]["shopping_list"], ["lait de coco en boîte : 1"])

    def test_meme_recette_deux_fois_remplace_au_lieu_d_additionner(self):
        """Run du 17/11 (multi-10) : pipérade ajoutée deux fois -> 20 œufs."""
        m = FauxModele([appel("ajouter_recette", recette="pipérade", personnes=4)],
                       [appel("ajouter_recette", recette="pipérade", personnes=2)], "Fait.")
        out = self.lancer(m, "Liste de la pipérade pour 2 personnes.")
        self.assertIn("œufs : 5", out["state"]["shopping_list"])
        self.assertIn("remplacées", " ".join(out["steps"][1]["outils"][0]["resultat"]["remarques"]))

    def test_remplacer_une_recette_garde_les_autres(self):
        m = FauxModele([appel("ajouter_recette", recette="pipérade"), appel("ajouter_recette", recette="pan bagnat")],
                       [appel("ajouter_recette", recette="pipérade", personnes=2)], "Fait.")
        out = self.lancer(m, "Liste de la pipérade pour 2 personnes et du pan bagnat.")
        liste = out["state"]["shopping_list"]
        self.assertIn("poivron vert : 2", liste)  # 1 (pipérade pour 2, arrondi) + 1 (pan bagnat)
        self.assertIn("thon : 250 g", liste)

    def test_seulement_entierement_ignore_n_ajoute_rien(self):
        """Run du 17/11 (multi-09) : seulement=[huile, ail] non demandés -> toute l'escalivada ajoutée une 2e fois."""
        m = FauxModele([appel("ajouter_recette", recette="Escalivada", personnes=4)],
                       [appel("ajouter_recette", recette="Escalivada", seulement=["Huile d'olive", "ail"])], "Fait.")
        out = self.lancer(m, "Prépare la liste de courses de l'escalivada pour 4 personnes.")
        self.assertFalse(out["steps"][1]["outils"][0]["resultat"]["ok"])
        self.assertIn("tomates : 8", out["state"]["shopping_list"])

    def test_produit_cherche_par_le_modele_seul_reste_redirige(self):
        """Run du 17/11 (multi-04) : le modèle cherche « vin blanc » de lui-même puis invente 250 ml."""
        reel = level2.executer_tool
        faux = lambda nom, args: {"ok": True, "produits": []} if nom == "chercher_produit" else reel(nom, args)  # noqa: E731
        m = FauxModele([appel("chercher_produit", nom="vin blanc")],
                       [appel("ajouter_a_la_liste", articles=[{"nom": "vin blanc", "quantite": 25, "unite": "cl"}])],
                       [appel("ajouter_recette", recette="Chili con carne", seulement=["vin blanc"])], "Fait.")
        with mock.patch.object(level2, "executer_tool", faux):
            out = self.lancer(m, "Ajoute à ma liste de courses le vin blanc du chili con carne, avec la quantité "
                                 "exprimée en millilitres.")
        self.assertIn("ajouter_recette", out["steps"][1]["outils"][0]["resultat"]["erreur"])
        self.assertEqual(out["state"]["shopping_list"], ["vin blanc sec : 200 ml"])

    def test_la_reponse_ne_pretend_pas_que_la_liste_est_envoyee(self):
        """Run du 17/11 (multi-02) : « La liste de courses a été envoyée » alors qu'elle attend le « oui »."""
        m = FauxModele([appel("ajouter_recette", recette="pipérade")], [appel("envoyer_liste")],
                       "La liste de courses a été envoyée.")
        out = self.lancer(m, "Prépare la liste de la pipérade puis envoie-la.")
        self.assertNotIn("a été envoyée", out["answer"])
        self.assertIn("réponds « oui »", out["answer"])
        self.assertTrue(out["state"]["pending_confirmation"])

    def test_un_envoi_rate_peut_etre_refait(self):
        """Run du 17/11 (multi-09) : envoyer_liste appelé quand la liste était vide, puis refusé comme « identique »."""
        m = FauxModele([appel("envoyer_liste"), appel("ajouter_recette", recette="escalivada")],
                       [appel("envoyer_liste")], [appel("envoyer_liste")], "Confirme.")
        out = self.lancer(m, "Liste de l'escalivada et envoie-la")
        self.assertFalse(out["steps"][0]["outils"][0]["resultat"]["ok"])  # liste encore vide
        self.assertTrue(out["state"]["pending_confirmation"])
        self.assertEqual(out["steps"][2]["outils"][0]["resultat"]["statut"], "deja_en_attente")

    def test_pas_de_je_ne_trouve_pas_quand_la_liste_est_faite(self):
        m = FauxModele([appel("ajouter_recette", recette="pad thaï", personnes=2)], level2.ABSTAIN)
        out = self.lancer(m, "Prépare la liste de courses du pad thaï pour 2 personnes.")
        self.assertTrue(out["answer"].startswith("Ta liste de courses"))

    def test_sortie_compatible_avec_eval(self):
        out = self.lancer(FauxModele("Bonjour."), "Bonjour")
        for cle in ["answer", "sources", "retrieved", "state", "steps", "tokens_in", "tokens_out", "latency"]:
            self.assertIn(cle, out)
        json.dumps(out, ensure_ascii=False)  # doit pouvoir s'écrire dans runs/*.jsonl


if __name__ == "__main__":
    unittest.main()
