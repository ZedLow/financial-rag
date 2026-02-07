---
title: Finance RAG Analyst
emoji: 📉
colorFrom: purple
colorTo: green
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

# Financial RAG Demo

This demo showcases a **constrained Financial RAG pipeline** designed to reduce hallucinations through **explicit routing and hard constraints**, not prompt tricks.

---

## What this demo does

- Routes queries based on detected company entities (Apple / Microsoft)
- Prevents accidental cross-company document mixing
- Processes financial tables as images to preserve structure
- Explicitly rejects unsupported or ambiguous queries

---

## How to test it

Try the following queries:

- `What was Apple’s total revenue in 2023?`
- `What is Microsoft’s operating income?`
- `Compare Apple and Microsoft revenues` → rejected or limited
- `What was Google’s revenue in 2023?` → rejected

The UI shows retrieved pages and scores to make the pipeline inspectable.

---

## Important limitations

- Explicit multi-company questions may trigger cross-entity reasoning
- Source-constrained prompts are not strictly enforced
- Dataset is intentionally small (demo-only)

For full technical details and design discussion, see the GitHub repository linked on the CV.