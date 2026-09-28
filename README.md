# dealpilot-hindsight
DealPilot: An AI Deal Intelligence Agent powered by Vectorize Hindsight persistent memory and Groq LLMs.


# DealPilot 🎯 — AI Deal Intelligence Agent with Hindsight Memory

DealPilot is an intelligent sales assistant powered by **Vectorize Hindsight** persistent memory and **Groq LLMs**. It eliminates manual CRM note digging and prevents lost deals by maintaining continuous, context-aware memory of every interaction across a complex deal cycle.

---

## 🚨 The Business Problem
Sales reps waste up to **30% of their week** re-reading old CRM notes, transcriptions, and emails before call prep. Worse, critical details fall through the cracks—like a CFO's objection to discounts, a promised security compliance doc, or specific stakeholder preferences. Generic chatbots treat every session as a blank slate, offering generic advice that damages deals.

---

## 💡 The Hindsight Advantage (Before vs. After Memory)

| Interaction | Scenario | Without Memory (Generic Bot) | With Hindsight Memory (DealPilot) |
| :--- | :--- | :--- | :--- |
| **Call 1** | Prospect mentions CFO hates upfront discounts and needs ISO 27001 docs. | Generates generic pitch. | Stores preferences, objections, and open commitments in Hindsight memory layer. |
| **Call 5** | Rep asks: *"Brief me for tomorrow's pricing call."* | *"Highlight ROI and offer a 10% discount to close fast."* ❌ | *"Do NOT lead with discounts (CFO objected in Call 1). Present ROI sheet built on their data and bring ISO 27001 compliance doc."* ✅ |

---

## 🏗️ System Architecture

```text
+-------------------+      User Query       +-------------------+
|  Sales Rep / CLI  | --------------------> |  DealPilot Agent  |
+-------------------+                       +-------------------+
                                                      |
                                   +------------------+------------------+
                                   |                                     |
                                   v                                     v
                        +----------------------+             +----------------------+
                        |   Groq LLM Engine    |             |   Hindsight Memory   |
                        |  (gpt-oss-120b /     | <---------> |  (Vectorize Cloud)   |
                        |   qwen3-32b)         |             |  Recall & Context    |
                        +----------------------+             +----------------------+
