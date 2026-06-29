"""
assistant.py — an LLM sales assistant that qualifies, advises and captures leads.

Pipeline:  retrieve (RAG)  ->  detect intent  ->  guardrailed LLM answer  ->  capture lead

Provider-agnostic: set LLM_PROVIDER=anthropic|openai and LLM_API_KEY.
Reference implementation behind production assistants like MAM Expert.
"""

import json
import os
import re

import requests

from knowledge import Retriever

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "claude-3-5-sonnet-latest"
                  if PROVIDER == "anthropic" else "gpt-4o-mini")

SYSTEM_PROMPT = """Esti asistentul de vanzari al unui magazin de scule si bricolaj.
Rolul tau: ajuti clientul sa gaseasca produsul potrivit si il conduci spre achizitie.

REGULI (guardrails) - nenegociabile:
- Foloseste DOAR informatiile din sectiunea CONTEXT de mai jos. Nu inventa
  niciodata preturi, stoc, specificatii sau produse.
- Daca nu ai informatia in CONTEXT, spune sincer ca verifici si oferi sa pui
  clientul in legatura cu un coleg.
- Fii scurt, prietenos si orientat spre urmatorul pas (rezervare / contact).
- Niciun sfat care depaseste produsele (medical, juridic etc.).
"""


def llm(messages: list[dict], max_tokens: int = 400) -> str:
    """Minimal provider-agnostic chat call (Anthropic or OpenAI)."""
    if not API_KEY:
        return "[LLM_API_KEY missing - set it to get live answers]"
    if PROVIDER == "anthropic":
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01"},
            json={"model": MODEL, "system": sys, "messages": turns,
                  "max_tokens": max_tokens},
            timeout=30,
        )
        return r.json()["content"][0]["text"]
    # openai-compatible
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": MODEL, "messages": messages, "max_tokens": max_tokens},
        timeout=30,
    )
    return r.json()["choices"][0]["message"]["content"]


def _norm(text: str) -> str:
    """Lowercase + strip Romanian diacritics (users type both ways)."""
    table = str.maketrans("ăâîșț", "aaist")
    return text.lower().translate(table)


def detect_intent(message: str) -> str:
    """Lightweight intent classification (rules first; cheap and explainable)."""
    m = _norm(message)
    if re.search(r"\b(rezerv|program|comand|cumpar|vreau sa)\b", m):
        return "buy"
    if re.search(r"\b(sun|telefon|whatsapp|coleg|contact)\b", m):
        return "book"
    if re.search(r"(vs|diferen|care e mai|mai bun)", m):
        return "compare"
    if re.search(r"(disponibil|stoc|pret|cat cost|aveti|recomand)", m):
        return "info"
    return "other"


def format_context(products: list[dict]) -> str:
    if not products:
        return "CONTEXT: (niciun produs relevant gasit)"
    lines = []
    for p in products:
        stock = "in stoc" if p["in_stock"] else "stoc epuizat"
        lines.append(f"- {p['name']} | {p['price_ron']} lei | {stock} | {p['description']}")
    return "CONTEXT (singura sursa de adevar):\n" + "\n".join(lines)


def capture_lead(message: str, intent: str, products: list[dict]) -> dict | None:
    """Turn a high-intent message into structured data for the CRM / call-center."""
    if intent not in ("buy", "book"):
        return None
    return {
        "intent": intent,
        "product": products[0]["id"] if products else None,
        "message": message,
        "contact_requested": intent == "book" or bool(products),
    }


def respond(message: str, retriever: Retriever) -> dict:
    products = retriever.search(message, k=3)
    intent = detect_intent(message)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": format_context(products)},
        {"role": "user", "content": message},
    ]
    answer = llm(messages)
    return {"answer": answer, "intent": intent,
            "lead": capture_lead(message, intent, products)}


def main() -> None:
    retriever = Retriever()
    print("Asistent vanzari (Ctrl+C pentru iesire)\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nLa revedere!")
            break
        if not msg:
            continue
        out = respond(msg, retriever)
        print(f"Assistant: {out['answer']}")
        if out["lead"]:
            print(f"[lead captured] {json.dumps(out['lead'], ensure_ascii=False)}")
        print()


if __name__ == "__main__":
    main()
