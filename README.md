# Nova: NovaCell Support Assistant (RAG chatbot)

A retrieval-augmented chatbot for Tier-1 telecom support questions, built to [PRD.md](PRD.md).
Every answer is generated only from three local knowledge sources and lists the documents it used.

| Piece | Choice |
|---|---|
| Knowledge | `data/faq.csv` (25), `data/tickets.db` (resolved only, 19), `data/telecom_guide.pdf` (600/100-char chunks) |
| Embeddings | `all-MiniLM-L6-v2`, run locally via ChromaDB's ONNX build (no torch) |
| Vector store | ChromaDB, persisted in `chroma_store/` (built automatically on first start) |
| Retrieval | Top-3 from each collection in parallel = 9 source-labelled docs |
| LLM | `qwen/qwen3.8-27b` on Groq (override with `GROQ_MODEL`), temperature 0, `reasoning_effort="none"` |
| UI | Streamlit (`app.py`) and a CLI (`main.py`); NovaCell theme in `.streamlit/config.toml` + CSS in `app.py` |

> FR-13 in the PRD names `qwen/qwen3.6-27b`; Groq no longer serves it, so the default is its successor.

## Refund agent ([agent_PRD.md](agent_PRD.md))

Every message is first classified as `requirement_inquiry` (existing RAG, unchanged) or
`refund_request` (Refund Agent). Both routes receive the last 4 exchanges as context; the
retrieval query is still the question alone.

```
User → route() ─┬─ requirement_inquiry → retrieve → grounded answer (+ history)
                └─ refund_request → Refund Agent
                     LLM tools: get_user_account, validate_refund  (user ID bound by the app)
                     code:      eligibility rules → ₹499 limit →
                                 ≤ ₹499: Confirm Refund / Cancel → process_refund (simulated)
                                 > ₹499: submit_to_support (no outcome is simulated)
```

| Rule | Where |
|---|---|
| Completed, ≤ 7 days, not already refunded, 0 < amount ≤ recharge | `validate_refund` in `refund_tools.py` |
| Which recharge a request applies to (exact amount match first) | `_select_recharge` in `refund_tools.py` |
| ₹499 approval limit, confirmation, execution | `refund_agent.py` (`process_refund` also re-checks both) |
| Only "Yes, proceed"-style replies confirm; hedged replies ask again | `parse_confirmation` in `refund_agent.py` |
| LLM-chosen amounts must be stated by the user or offered by the agent | grounding guard in `refund_agent.py` |

The demo user (`user_123`, Rahul) has two recharges, ₹499 and ₹999; refunds update a per-session
copy, shown in the sidebar, with **Reset demo account** to start over. If the LLM is unavailable
(rate limit, no key), intent detection falls back to keyword rules and the agent to deterministic
amount parsing, so refunds still work. Each answer has an **Agent steps** panel with the logged
workflow (`[Intent] …`, `[Agent] validate_refund(amount=499)`, …); no model reasoning is shown.

## Run locally (Windows)

Python 3.11+ in a project-local `.venv` (built and tested on 3.13).

```bash
uv venv .venv --python 3.13
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY` (free at console.groq.com).

```bash
.venv/Scripts/streamlit run app.py
```

```bash
.venv/Scripts/python main.py
```

On the very first start the app downloads the ~80 MB embedding model and builds `chroma_store/` from `data/` (about 30 s). Later starts take a few seconds.

## Deploy: GitHub + Streamlit Community Cloud

1. Push this folder to a GitHub repository. `.gitignore` already keeps out `.env`, `.venv`, `chroma_store/` and logs.
2. At share.streamlit.io, sign in with GitHub, click **Create app**, choose the repo, branch `main` and main file `app.py`.
3. Under **Advanced settings**, pick Python **3.13** and paste the secret:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

   Streamlit exposes top-level secrets as environment variables, so no code change is needed.
4. Deploy. The first boot builds the knowledge base from `data/`; you then get a public `*.streamlit.app` link.

Notes: the files in `data/` become part of the repo (use a private repo if they shouldn't be public). All visitors share your Groq rate limit (free tier: about 2–3 answers a minute). Free apps sleep when idle and take a moment to wake.

## Updating knowledge

Edit files in `data/`, then rebuild locally:

```bash
.venv/Scripts/python ingest_all.py
```

On Streamlit Cloud, push the changed data and use **Reboot app**; `chroma_store/` is rebuilt on start because it isn't stored in git. Each `ingest_*.py` can also be run on its own; re-running rebuilds that collection from scratch (US-06, US-07).

## Tests (no API key needed)

```bash
.venv/Scripts/python -m pytest tests -q
```

The tests swap Groq for fake models and check ingestion counts, retrieval, prompt wiring, the Streamlit flow, all seven refund demo scenarios and the refund guardrails (`tests/test_agent.py`).

## Files

| File | Role |
|---|---|
| `config.py` | Paths, model names, embeddings, shared Chroma client, auto-ingest of empty collections |
| `ingest_faq.py`, `ingest_tickets.py`, `ingest_guides.py`, `ingest_all.py` | Build the three collections |
| `retriever.py` | Parallel merged retriever; register new collections in `COLLECTIONS` (NFR-06) |
| `chain.py` | Grounding prompt, shared Groq config, conversation history, streaming, logging |
| `assistant.py` | Routes each message to RAG or the Refund Agent |
| `intent.py` | LLM intent classifier (structured output) with keyword fallback |
| `refund_agent.py` | Refund Agent: tool loop, state machine, approval limit, confirmation |
| `refund_tools.py` | `get_user_account`, `validate_refund`, `process_refund`, `submit_to_support` |
| `refund_data.py` | Hard-coded demo account and per-session store |
| `app.py` | Streamlit chat: topic cards, sample questions, streamed answers, Sources panel, 👍/👎 |
| `main.py` | CLI REPL; `quit` exits |
| `logs/interactions.jsonl` | One line per answer (question, answer, sources, fallback flag) and per feedback click |

## Grounding rules (in `chain.py`)

- Answer only from retrieved context; never state prices, fees or policies that are not in it.
- If the context doesn't answer it, reply with a fixed fallback that points to 611 or the NovaCell app.
- No access to account data; give the self-serve path (e.g. `*123#`) instead.
- Ticket resolutions are used for troubleshooting steps only. Compensation or credits given in one case are never promised.
- Cite sources inline, e.g. `[FAQ #2]`, `[TK-008]`.
