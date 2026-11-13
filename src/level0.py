"""Niveau 0 : LLM seul. Un prompt soigné, ni documents ni outils.

C'est la référence à battre : les niveaux 1 et 2 doivent prouver ce qu'ils apportent
par rapport à ce score.

Contrat commun à tous les niveaux : answer(question) -> dict avec les clés ci-dessous.
eval.py ne lit que ces clés, donc ajouter ce que vous voulez en plus (il finira dans la trace).
"""
from src.llm import chat

SYSTEM = (
    "Tu es un chef anti-gaspi. Tu aides l'utilisateur à cuisiner avec ce qu'il a dans son "
    "frigo et ses placards, en limitant le gaspillage. Réponds en français, de façon concise. "
    "Si tu n'es pas sûr d'une information, ou si la question n'a rien à voir avec la cuisine, "
    "dis-le clairement au lieu d'inventer. Si la demande est trop vague, pose une question de précision."
)


def answer(question: str) -> dict:
    r = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ])
    return {
        "answer": r["message"].get("content") or "",
        "sources": [],        # niveau 1+ : documents cités
        "retrieved": None,    # niveau 1+ : liste ordonnée des ids de documents retrouvés
        "state": {},          # niveau 2+ : état final (ex. {"shopping_list": [...], "sent": False})
        "steps": [],          # niveau 2+ : trace (outil, arguments, résultat)
        "tokens_in": r["tokens_in"],
        "tokens_out": r["tokens_out"],
        "latency": r["latency"],
    }
