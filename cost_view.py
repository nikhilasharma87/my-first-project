"""'Token costs' tab of the Insights dashboard: an interactive version of the
RAG Cost Estimation workbook (Inference, Embedding and Cost Savings sheets).

Every assumption is an input. "Measured from Nova" pre-fills the inputs from the
conversation log and the knowledge base; "Workbook assumptions" pre-fills the
workbook's numbers. Either way any value can be overwritten on the page.
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

import costs as C
import insights as I

NAVY, ORANGE_UI = "#0A3A8C", "#F26B1D"
SERIES_BLUE, SERIES_ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0B1F3A", "#52606F", "#E6EAF0"

MEASURED, WORKBOOK = "Measured from Nova", "Workbook assumptions"


@st.cache_data(ttl=300, show_spinner="Measuring the knowledge base…")
def _kb() -> pd.DataFrame:
    return C.kb_documents()


@st.cache_data(show_spinner=False)
def _prompts() -> dict:
    return C.prompt_words()


def _money(x: float, small: bool = False) -> str:
    if small:
        return f"${x:,.5f}"
    return f"${x:,.2f}"


def _tiles(items: list[tuple[str, str, str]]) -> None:
    for col, (label, value, sub) in zip(st.columns(len(items)), items):
        col.html(f"<div class='ni-tile'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>")


def _h(title: str, sub: str) -> None:
    st.html(f"<div class='ni-h'>{title}</div><div class='ni-sub'>{sub}</div>")


def render(df: pd.DataFrame, period_days: int | None, axis_cfg) -> None:
    kb, prompts = _kb(), _prompts()

    st.html("<div class='ni-sub' style='margin-top:4px'>A live version of the <b>RAG Cost Estimation</b> "
            "workbook. Start from what Nova actually sends to the model, or from the workbook's assumptions, "
            "then change any number. Uses the period and channel filters above.</div>")

    c1, c2, c3 = st.columns([1.4, 1.2, 1.2])
    mode = c1.segmented_control("Start from", [MEASURED, WORKBOOK], default=MEASURED, key="cost_mode",
                                help="Pre-fills the inputs below. You can still edit every value.") or MEASURED
    tpw = c2.slider("Tokens per word", 1.0, 2.0, C.TOKENS_PER_WORD, 0.05, key="cost_tpw",
                    help="Converts measured word counts to tokens. The workbook uses 1.3; "
                         "check it against Groq's real counts shown below.")
    with_classifier = c3.toggle("Include intent-classifier call", value=mode == MEASURED, key=f"cost_cls_{mode}",
                                help="Nova makes a second, small LLM call per message to decide between "
                                     "a support answer and the refund agent. The workbook leaves it out.")

    m = C.measure(df, kb, prompts)
    prof = C.profile_table(m, tpw)

    # ------------------------------------------------------------ 1. tokens per query
    _h("1 · Tokens per query", "Edit the <b>Used</b> column. Measured values come from the source shown.")
    prof["used"] = prof["measured"] if mode == MEASURED else prof["workbook"]
    left, right = st.columns([1.35, 1])
    with left:
        edited = st.data_editor(
            prof[["component", "used", "workbook", "measured", "source"]], hide_index=True,
            use_container_width=True, key=f"cost_profile_{mode}_{tpw}_{len(df)}",
            disabled=["component", "workbook", "measured", "source"],
            column_config={
                "component": st.column_config.TextColumn("Part of the prompt"),
                "used": st.column_config.NumberColumn("Used", min_value=0, step=1, format="%d",
                                                      help="Tokens per query used in every cost below"),
                "workbook": st.column_config.NumberColumn("Workbook", format="%d"),
                "measured": st.column_config.NumberColumn("Measured", format="%d"),
                "source": st.column_config.TextColumn("Measured from", width="medium"),
            })
    used = dict(zip(edited["component"], edited["used"].fillna(0)))
    rag_input = sum(used[c] for c in C.INPUT_PARTS)
    rag_output = used["Output (answer)"]
    cls_in = round(m.classifier_input_words * tpw) if with_classifier else 0
    cls_out = C.CLASSIFIER_OUTPUT_TOKENS if with_classifier else 0
    input_tokens, output_tokens = rag_input + cls_in, rag_output + cls_out

    with right:
        parts = pd.DataFrame({"part": C.INPUT_PARTS + (["Intent classifier"] if with_classifier else []),
                              "tokens": [used[c] for c in C.INPUT_PARTS] + ([cls_in] if with_classifier else [])})
        parts = parts[parts["tokens"] > 0]
        parts["share"] = parts["tokens"] / parts["tokens"].sum()
        bars = alt.Chart(parts).mark_bar(size=18, color=SERIES_BLUE, cornerRadiusEnd=4).encode(
            y=alt.Y("part:N", sort="-x", title=None,
                    axis=alt.Axis(grid=False, labelLimit=180, labelOverlap=False, labelPadding=8)),
            x=alt.X("tokens:Q", title="Input tokens per query"),
            tooltip=[alt.Tooltip("part:N", title="Part"), alt.Tooltip("tokens:Q", title="Tokens", format=","),
                     alt.Tooltip("share:Q", title="Share of input", format=".0%")])
        labels = alt.Chart(parts).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
            y=alt.Y("part:N", sort="-x"), x="tokens:Q", text=alt.Text("tokens:Q", format=","))
        st.altair_chart(axis_cfg((bars + labels).properties(height=32 * len(parts) + 30)),
                        use_container_width=True)

    real = (f"Groq reported {m.real_input:,.0f} in / {m.real_output:,.0f} out on {m.real_count} answers"
            if m.real_count else "Groq's real counts are logged from now on")
    _tiles([
        ("Input tokens per query", f"{input_tokens:,.0f}",
         f"answer {rag_input:,.0f}" + (f" + classifier {cls_in:,.0f}" if with_classifier else "")),
        ("Output tokens per query", f"{output_tokens:,.0f}",
         f"answer {rag_output:,.0f}" + (f" + classifier {cls_out}" if with_classifier else "")),
        ("Total tokens per query", f"{input_tokens + output_tokens:,.0f}", f"workbook: 2,790"),
        ("Real usage check", f"{m.real_input + m.real_output:,.0f}" if m.real_count else "–", real),
    ])

    # ------------------------------------------------------------ 2. volume
    _h("2 · Volume", "How many questions reach the chatbot")
    msgs = len(df)
    span = period_days or (max((df["ts"].max() - df["ts"].min()).days, 1) if msgs else 1)
    current = msgs / span if msgs else 0
    v1, v2, v3 = st.columns(3)
    qpd = v1.number_input("Queries per day", min_value=0, value=C.WORKBOOK_QUERIES_PER_DAY, step=500,
                          key="cost_qpd", help=f"Nova currently handles about {current:,.1f} messages a day "
                                               "in the selected period. The workbook assumes 5,000.")
    days = v2.number_input("Days per month", min_value=1, max_value=31, value=C.WORKBOOK_DAYS, key="cost_days")
    qpm = qpd * days
    v3.html(f"<div class='ni-tile'><div class='l'>Queries per month</div><div class='v'>{qpm:,.0f}</div>"
            f"<div class='s'>Nova today: about {current * days:,.0f} a month</div></div>")

    # ------------------------------------------------------------ 3. inference
    _h("3 · Inference cost per model", "$ per 1M tokens. Edit prices or add rows (for example your Groq "
       "paid-tier price). Cost per query = tokens × price ÷ 1M.")
    m1, m2 = st.columns([1, 1.35])
    with m1:
        models = st.data_editor(
            C.WORKBOOK_MODELS, num_rows="dynamic", hide_index=True, use_container_width=True, key="cost_models",
            column_config={
                "model": st.column_config.TextColumn("Model", required=True),
                "input_per_1m": st.column_config.NumberColumn("Input $/1M", min_value=0.0, format="$%.2f"),
                "output_per_1m": st.column_config.NumberColumn("Output $/1M", min_value=0.0, format="$%.2f"),
            })
        models = models.dropna(subset=["model"])
        inf = C.inference_costs(input_tokens, output_tokens, models, qpm)
        names = inf["model"].tolist()
        chosen = st.selectbox("Model for the savings calculation", names,
                              index=names.index("GPT-6.2 Sol") if "GPT-6.2 Sol" in names else 0,
                              key="cost_model") if names else None
    with m2:
        if inf.empty:
            st.info("Add a model with input and output prices.")
        else:
            inf["highlight"] = inf["model"].eq(chosen).map({True: "Selected", False: "Other"})
            chart = alt.Chart(inf).mark_bar(size=18, cornerRadiusEnd=4).encode(
                y=alt.Y("model:N", sort="-x", title=None,
                        axis=alt.Axis(grid=False, labelLimit=200, labelOverlap=False, labelPadding=8)),
                x=alt.X("cost_per_month:Q", title="Inference cost per month ($)", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("highlight:N", legend=None,
                                scale=alt.Scale(domain=["Selected", "Other"], range=[SERIES_ORANGE, SERIES_BLUE])),
                tooltip=[alt.Tooltip("model:N", title="Model"),
                         alt.Tooltip("cost_per_query:Q", title="Per query", format="$,.5f"),
                         alt.Tooltip("cost_per_month:Q", title="Per month", format="$,.2f")])
            labels = alt.Chart(inf).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
                y=alt.Y("model:N", sort="-x"), x="cost_per_month:Q",
                text=alt.Text("cost_per_month:Q", format="$,.2f"))
            st.altair_chart(axis_cfg((chart + labels).properties(height=32 * len(inf) + 30)),
                            use_container_width=True)
    if not inf.empty:
        st.dataframe(inf[["model", "input_cost_per_query", "output_cost_per_query", "cost_per_query",
                          "cost_per_month"]], hide_index=True, use_container_width=True,
                     column_config={
                         "model": "Model",
                         "input_cost_per_query": st.column_config.NumberColumn("Input $/query", format="$%.5f"),
                         "output_cost_per_query": st.column_config.NumberColumn("Output $/query", format="$%.5f"),
                         "cost_per_query": st.column_config.NumberColumn("Total $/query", format="$%.5f"),
                         "cost_per_month": st.column_config.NumberColumn("Total $/month", format="$%.2f"),
                     })

    # ------------------------------------------------------------ 4. embedding
    _h("4 · Embedding cost", "Indexing the knowledge base plus embedding every question. Nova runs "
       "all-MiniLM-L6-v2 locally, so its real embedding cost is $0; set a price to cost a paid API instead.")
    kb_measured = round(kb["words"].sum() * tpw) if len(kb) else 0
    e1, e2, e3 = st.columns(3)
    kb_tokens = e1.number_input("Knowledge-base tokens", min_value=0,
                                value=kb_measured if mode == MEASURED else 5_551_000, step=1000,
                                key=f"cost_kb_{mode}_{tpw}",
                                help=f"Nova's KB: {len(kb)} chunks, {kb_measured:,} tokens. "
                                     "Workbook: 5,551,000 (10,000 tickets, 50 FAQs, 50 PDFs).")
    reindex = e2.number_input("Re-indexes per month", min_value=0.0, value=1.0, step=1.0, key="cost_reindex",
                              help="The workbook counts one full embedding run per month.")
    embed_price = e3.number_input("Embedding $ per 1M tokens", min_value=0.0,
                                  value=0.0 if mode == MEASURED else C.WORKBOOK_EMBED_PRICE, step=0.01,
                                  format="%.2f", key=f"cost_embed_{mode}")
    emb = C.embedding_costs(kb_tokens, used["User query"], qpm, embed_price, reindex)
    _tiles([
        ("Knowledge base", _money(emb["kb_cost"]), f"{kb_tokens * reindex:,.0f} tokens a month"),
        ("User queries", _money(emb["query_cost"]), f"{emb['query_tokens']:,.0f} tokens a month"),
        ("Total embedding", _money(emb["total"]), "per month"),
    ])

    # ------------------------------------------------------------ 5. savings
    _h("5 · Savings vs human agents", "Cost Savings sheet. Nova does not answer everything: questions it "
       "can't answer still go to an agent, so the slider sends that share back to humans.")
    k = I.kpis(df) if msgs else {"answered_rate": None}
    answered = k.get("answered_rate")
    s1, s2, s3, s4 = st.columns(4)
    per_agent = s1.number_input("Queries per agent per day", min_value=1, value=600, step=50, key="cost_agent_q")
    salary = s2.number_input("Salary per agent / month ($)", min_value=0.0, value=400.0, step=50.0, key="cost_salary")
    admin = s3.number_input("Admin per agent / month ($)", min_value=0.0, value=100.0, step=10.0, key="cost_admin")
    default_share = round(answered * 100) if (mode == MEASURED and answered is not None) else 100
    share = s4.slider("Share Nova resolves (%)", 0, 100, default_share, key=f"cost_share_{mode}_{default_share}",
                      help=(f"Measured answered rate: {answered:.0%}. " if answered is not None else "")
                           + "The workbook assumes 100%.")

    inference_month = float(inf.loc[inf["model"] == chosen, "cost_per_month"].iloc[0]) if chosen else 0.0
    chatbot = inference_month + emb["total"]
    sav = C.savings(qpd, days, per_agent, salary, admin, chatbot, share / 100)
    _tiles([
        ("Agents needed", f"{sav['agents_without_bot']:.1f} → {sav['agents_with_bot']:.1f}",
         "without → with Nova"),
        ("Human-only cost", _money(sav["human_only"]), "per month"),
        ("With Nova", _money(sav["with_bot"]), f"{chosen or '–'} + embedding + remaining agents"),
        ("Savings", f"{_money(sav['savings'])} ({sav['savings_pct']:.0%})", "per month"),
    ])

    comp = pd.DataFrame([
        {"scenario": "Human agents only", "part": "Human agents", "cost": sav["human_only"]},
        {"scenario": "With Nova", "part": "Human agents", "cost": sav["agents_with_bot"] * sav["cost_per_agent"]},
        {"scenario": "With Nova", "part": "Nova (LLM + embedding)", "cost": chatbot},
    ])
    bars = alt.Chart(comp).mark_bar(size=22, stroke="#FFFFFF", strokeWidth=2).encode(
        y=alt.Y("scenario:N", title=None, sort=["Human agents only", "With Nova"],
                axis=alt.Axis(grid=False, labelOverlap=False, labelPadding=8)),
        x=alt.X("sum(cost):Q", title="Cost per month ($)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("part:N", title=None, scale=alt.Scale(domain=["Human agents", "Nova (LLM + embedding)"],
                                                               range=[SERIES_BLUE, SERIES_ORANGE])),
        tooltip=[alt.Tooltip("scenario:N", title="Scenario"), alt.Tooltip("part:N", title="Part"),
                 alt.Tooltip("cost:Q", title="Cost", format="$,.2f")])
    st.altair_chart(axis_cfg(bars.properties(height=120)), use_container_width=True)

    # ------------------------------------------------------------ export
    scenario = {
        "start_from": mode, "tokens_per_word": tpw, "include_classifier": with_classifier,
        "tokens_per_query": {**{c: float(v) for c, v in used.items()},
                             "classifier_input": cls_in, "classifier_output": cls_out,
                             "total_input": float(input_tokens), "total_output": float(output_tokens)},
        "queries_per_day": qpd, "days_per_month": days, "queries_per_month": qpm,
        "models": inf.drop(columns=["highlight"], errors="ignore").to_dict("records"),
        "embedding": {"price_per_1m": embed_price, "reindexes_per_month": reindex, **emb},
        "savings": {"model": chosen, "share_resolved": share / 100, "queries_per_agent_per_day": per_agent,
                    "salary": salary, "admin": admin, **sav},
    }
    st.download_button("Download this scenario (JSON)", json.dumps(scenario, indent=2, default=float),
                       file_name="nova_cost_scenario.json", mime="application/json", icon=":material/download:")
    with st.expander("How the numbers are measured", icon=":material/info:"):
        st.markdown(
            "- **Tokens** are word counts × *tokens per word*. Nova's model (Qwen on Groq) uses its own "
            "tokenizer, so compare with **Real usage check**: Groq's own counts, logged for each answer from now on.\n"
            "- **System prompt**: the grounding prompt in `chain.py` without the retrieved context.\n"
            "- **FAQs / Tickets / Guides**: the documents retrieved for each logged answer (top-3 per source); "
            "with no answers logged, 3 × the average document size.\n"
            "- **Conversation history**: the previous 4 exchanges that Nova sends with each answer.\n"
            "- **Output**: real answers only; demo data has placeholder answers and fallbacks are excluded.\n"
            "- **Intent classifier**: one extra call per message (classifier prompt + history + question in, "
            f"about {C.CLASSIFIER_OUTPUT_TOKENS} tokens out).\n"
            "- Model prices and the savings inputs are the workbook's; replace them with real quotes.")
