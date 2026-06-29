# LLM Sales Assistant 🛒🤖

A reference implementation of the **conversational AI sales assistant** architecture I ship to production — the same approach behind **MAM Expert**, a live assistant that has handled **12,300+ customer messages at 99.92% accuracy**.

It's not a generic chatbot that deflects tickets. It's an assistant that **qualifies, advises and moves the customer toward a purchase** — with guardrails so it never invents facts.

---

## 📈 Case study: MAM Expert (production)
Built for a DIY / hardware retailer that was losing inquiries every evening and weekend.

| Metric | Result |
|---|---|
| Messages handled | **12,300+** |
| Accuracy | **99.92%** |
| Conversations | **2,154** |
| Clear purchase intent | **46.8%** |
| Attributed commercial value | **45,000–60,000 RON / 2.5 months** |
| ROI | **2.5–3.5×** |

**3 engineering lessons that made it work**
1. **Guardrails first.** The model informs and routes; it never invents prices or stock. One clean source of truth beats a clever prompt.
2. **Sell, don't deflect.** Intent detection + qualification turn questions into routed, high-intent leads.
3. **Structured handoff.** Every booking/lead is captured as structured data the team can act on.

---

## 🏗 Architecture
```
User message
   │
   ▼
1. Retrieve relevant product knowledge (RAG)
2. Detect intent + qualify the lead
3. LLM answer with guardrails (grounded only in the knowledge base)
4. Capture booking / lead as structured JSON
   │
   ▼
Structured response  →  CRM / call-center / dashboard
```

## ✨ What this reference implementation shows
- **Retrieval-augmented answers** grounded in a product knowledge base (no hallucinated prices/stock).
- **Intent detection & lead qualification** (info / compare / buy / book).
- **Guardrailed system prompt** (provider-agnostic LLM call: Claude / OpenAI).
- **Structured lead capture** (intent, product, contact) as JSON ready for a CRM.

## 🚀 Run it
```bash
pip install -r requirements.txt
export LLM_API_KEY="your_key"        # Claude or OpenAI
python assistant.py
```

## 🧱 Tech
`Python` · LLM APIs (Claude / OpenAI) · RAG (bag-of-words + cosine; swap in embeddings) · prompt engineering · structured outputs · guardrails

## 📁 Files
- `assistant.py` — the assistant (retrieval → intent → guardrailed answer → lead capture)
- `knowledge.py` — dependency-free vector search over `products.json`
- `products.json` — sample product catalog
- `requirements.txt`

---
*Built by [David Mangău](https://www.linkedin.com/in/david-mangău-9578602a9) — AI Engineer. Reference implementation; the production system (MAM Expert) is live and client-owned.*
