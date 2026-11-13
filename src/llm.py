"""Client LLM unique pour tous les niveaux.

Parle le protocole « OpenAI-compatible » (/chat/completions). Ollama l'expose en
local sur http://localhost:11434/v1, donc passer du modèle local à l'API du prof
= changer 3 variables dans .env, sans toucher au code.

Mettre LLM_MODEL=mock pour tester la chaîne d'évaluation sans aucun modèle.
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")


def chat(messages, tools=None, temperature=0.0, max_tokens=512, model=None, seed=0):
    """Un appel au modèle.

    Retourne {"message": dict, "tokens_in": int, "tokens_out": int, "latency": float}.
    `message` suit le format OpenAI : {"role", "content", "tool_calls"?}.
    """
    model = model or MODEL
    if model == "mock":
        return _mock(messages)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if tools:
        payload["tools"] = tools

    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage") or {}
    return {
        "message": data["choices"][0]["message"],
        "tokens_in": usage.get("prompt_tokens", 0),
        "tokens_out": usage.get("completion_tokens", 0),
        "latency": time.time() - t0,
    }


def _mock(messages):
    """Faux modèle déterministe, uniquement pour tester eval.py hors ligne."""
    time.sleep(0.01)
    question = messages[-1]["content"]
    return {
        "message": {"role": "assistant", "content": f"[mock] Réponse simulée à : {question[:60]}"},
        "tokens_in": sum(len(m["content"]) for m in messages) // 4,
        "tokens_out": 20,
        "latency": 0.01,
    }
