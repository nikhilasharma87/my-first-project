"""Nova insights: what customers ask, how well Nova answers, and what to improve
in the knowledge base. Reads the conversation log; nothing here calls the LLM.

Open from the chat sidebar ("Insights dashboard") or /dashboard.
"""
import altair as alt
import pandas as pd
import streamlit as st

import insights as I
from config import INTERACTION_LOG

st.set_page_config(page_title="Nova Insights", page_icon="📊", layout="wide")

# Brand colours for chrome; chart series use the validated reference pair
# (dataviz palette slots 1 and 2: blue #2a78d6, orange #eb6834).
NAVY, ORANGE_UI = "#0A3A8C", "#F26B1D"
SERIES_BLUE, SERIES_ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0B1F3A", "#52606F", "#E6EAF0"

st.html(f"""
<style>
  .block-container {{ padding-top: 1.6rem; max-width: 1240px; }}
  .ni-hero {{ background: {NAVY}; border-bottom: 4px solid {ORANGE_UI}; border-radius: 14px;
             padding: 18px 22px; color: #fff; margin-bottom: 14px; }}
  .ni-hero h1 {{ color:#fff; font-size: 1.55rem; margin: 2px 0 2px; padding: 0; }}
  .ni-hero p {{ margin: 0; opacity: .85; font-size: .92rem; }}
  .ni-logo {{ font-weight: 800; font-size: .95rem; }} .ni-logo b {{ color: {ORANGE_UI}; }}
  .ni-tile {{ background: #fff; border: 1px solid {GRID}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
  .ni-tile .l {{ color: {INK_2}; font-size: .82rem; }}
  .ni-tile .v {{ color: {INK}; font-size: 1.75rem; font-weight: 650; line-height: 1.25; }}
  .ni-tile .s {{ color: {INK_2}; font-size: .78rem; }}
  .ni-h {{ color: {NAVY}; font-weight: 700; font-size: 1.12rem; margin: 18px 0 2px; }}
  .ni-sub {{ color: {INK_2}; font-size: .86rem; margin-bottom: 6px; }}
  .ni-act {{ background:#fff; border:1px solid {GRID}; border-left: 4px solid {ORANGE_UI}; border-radius: 0;
            padding: 10px 14px; height: 100%; margin-bottom: 10px; }}
  .ni-act .n {{ font-size: 1.4rem; font-weight: 650; color: {INK}; }}
  .ni-act .t {{ font-size: .84rem; color: {INK_2}; }}
  .ni-demo {{ background: #FFF6F0; border: 1px solid {ORANGE_UI}; border-radius: 10px; padding: 8px 12px;
             color: {INK}; font-size: .86rem; }}
</style>
""")

st.html("<div class='ni-hero'><div class='ni-logo'>Nova<b>Cell</b> · Insights</div>"
        "<h1>What customers ask Nova, and what to fix next</h1>"
        "<p>Built from Nova's conversation log. Topics come from the knowledge-base article that best "
        "matched each question.</p></div>")

# ---------------------------------------------------------------- filters (one row)
f1, f2, f3, f4 = st.columns([1.1, 1.6, 1.1, 1])
period = f1.selectbox("Period", ["Last 7 days", "Last 14 days", "Last 30 days", "All time"], index=1)
include_demo = f3.toggle("Include demo data", value=I.DEMO_LOG.exists(),
                         help="Demo conversations from scripts/seed_demo_data.py, labelled channel 'demo'.")
f4.page_link("app.py", label="Back to chat", icon=":material/chat:")


@st.cache_data(ttl=30, show_spinner=False)
def _load(include_demo: bool, mtimes: tuple):
    return I.load_log(include_demo=include_demo), I.load_catalogue()


mtimes = tuple(p.stat().st_mtime if p.exists() else 0 for p in (INTERACTION_LOG, I.DEMO_LOG))
log, catalogue = _load(include_demo, mtimes)

channels_all = sorted(log["channel"].unique()) if not log.empty else []
channels = f2.multiselect("Channel", channels_all, default=channels_all)
days = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}.get(period)
df = I.filter_log(log, days, channels) if not log.empty else log

if include_demo and "demo" in channels and (log["channel"] == "demo").any():
    st.html("<div class='ni-demo'><b>Includes demo data.</b> Demo conversations use the real retriever and "
            "refund agent, but they are not real customers. Untick <i>Include demo data</i> to see only real "
            "conversations.</div>")


def _axis_cfg(chart):
    return (chart.configure_view(stroke=None)
            .configure_axis(gridColor=GRID, gridWidth=1, domainColor=GRID, tickColor=GRID,
                            labelColor=INK_2, titleColor=INK_2, labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=INK, titleColor=INK_2, orient="top", symbolType="square"))


def render_conversations(df: pd.DataFrame) -> None:
    # ---------------------------------------------------------------- KPI tiles
    k = I.kpis(df)
    pct = lambda x: "–" if x is None else f"{x:.0%}"  # noqa: E731
    tiles = [
        ("Conversations", f"{k['conversations']:,}", f"{k['messages']:,} messages"),
        ("Answered by Nova", pct(k["answered_rate"]), f"{k['unanswered']} sent to 611 without an answer"),
        ("Refund requests", f"{k['refund_requests']:,}", "messages handled by the refund agent"),
        ("Rated helpful", pct(k["helpful_rate"]), f"of {k['ratings']} rated answers"),
    ]
    for col, (label, value, sub) in zip(st.columns(4), tiles):
        col.html(f"<div class='ni-tile'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='s'>{sub}</div></div>")




    # ---------------------------------------------------------------- what customers ask
    left, right = st.columns([1.35, 1])
    topics = I.by_topic(df)
    with left:
        st.html("<div class='ni-h'>What customers ask about</div>"
                "<div class='ni-sub'>Messages per topic, split by whether Nova had a verified answer</div>")
        long = topics.melt(id_vars=["topic", "messages", "answered_pct"], value_vars=["answered", "not_answered"],
                           var_name="status", value_name="count")
        long["status"] = long["status"].map({"answered": "Answered", "not_answered": "Not answered"})
        order = topics["topic"].tolist()
        bars = alt.Chart(long).mark_bar(size=18, stroke="#FFFFFF", strokeWidth=2).encode(
            y=alt.Y("topic:N", sort=order, title=None,
                    axis=alt.Axis(grid=False, labelLimit=180, labelOverlap=False, labelPadding=8)),
            x=alt.X("sum(count):Q", title="Messages", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("status:N", title=None,
                            scale=alt.Scale(domain=["Answered", "Not answered"], range=[SERIES_BLUE, SERIES_ORANGE])),
            order=alt.Order("status:N", sort="ascending"),
            tooltip=[alt.Tooltip("topic:N", title="Topic"), alt.Tooltip("status:N", title="Status"),
                     alt.Tooltip("count:Q", title="Messages"), alt.Tooltip("messages:Q", title="Topic total"),
                     alt.Tooltip("answered_pct:Q", title="Answered", format=".0%")],
        ).properties(height=alt.Step(32))  # one 32px row per topic; bars are 18px, the rest is air
        st.altair_chart(_axis_cfg(bars), use_container_width=True)
        with st.expander("View data", icon=":material/table:"):
            st.dataframe(topics[["topic", "messages", "conversations", "answered", "not_answered", "answered_pct",
                                 "thumbs_down", "avg_match"]],
                         hide_index=True, use_container_width=True,
                         column_config={"answered_pct": st.column_config.NumberColumn("answered", format="percent"),
                                        "avg_match": st.column_config.NumberColumn("avg. match", format="%.2f")})

    with right:
        st.html("<div class='ni-h'>Messages per day</div><div class='ni-sub'>All routes, selected channels</div>")
        daily = I.daily_volume(df)
        daily["date"] = pd.to_datetime(daily["date"])
        base = alt.Chart(daily).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", grid=False, tickCount=7)),
            y=alt.Y("messages:Q", title="Messages", axis=alt.Axis(tickMinStep=1)),
        )
        line = base.mark_line(color=SERIES_BLUE, strokeWidth=2, strokeCap="round", strokeJoin="round")
        pts = base.mark_point(color=SERIES_BLUE, filled=True, size=70, stroke="#FFFFFF", strokeWidth=2, opacity=1).encode(
            tooltip=[alt.Tooltip("date:T", title="Day", format="%a %d %b"), alt.Tooltip("messages:Q", title="Messages")])
        st.altair_chart(_axis_cfg((line + pts).properties(height=260)), use_container_width=True)

        st.html("<div class='ni-h'>Most asked questions</div>")
        tq = I.top_questions(df, 10)
        st.dataframe(tq, hide_index=True, use_container_width=True, height=300,
                     column_order=["times_asked", "answered_pct", "question", "topic", "thumbs_down"],
                     column_config={
                         "question": st.column_config.TextColumn("Question", width="large"),
                         "times_asked": st.column_config.NumberColumn("Asked"),
                         "topic": st.column_config.TextColumn("Topic"),
                         "answered_pct": st.column_config.ProgressColumn("Answered", format="percent",
                                                                         min_value=0, max_value=1),
                         "thumbs_down": st.column_config.NumberColumn("👎"),
                     })

    # ---------------------------------------------------------------- knowledge base refinement
    st.html("<div class='ni-h'>Improve the knowledge base</div>"
            "<div class='ni-sub'>Actions for the support team, most impactful first. "
            "Each is backed by the conversations shown.</div>")
    recs = I.recommendations(df, catalogue)
    if recs.empty:
        st.success("No knowledge-base issues found for this period.")
    else:
        summary = recs.groupby("action").agg(items=("item", "count"), messages=("messages", "sum"))
        action_order = ["Write a new FAQ", "Turn a support ticket into an FAQ", "Review an unhelpful answer",
                        "Strengthen existing content", "Fix outdated wording", "Review unused content"]
        shown = [a for a in action_order if a in summary.index]
        for start in range(0, len(shown), 3):
            for col, action in zip(st.columns(3), shown[start:start + 3]):
                row = summary.loc[action]
                msgs = int(row["messages"])
                col.html(f"<div class='ni-act'><div class='t'><b>{action}</b></div>"
                         f"<div class='n'>{row['items']}</div>"
                         f"<div class='t'>{'no' if msgs == 0 else f'{msgs:,}'} customer message{'' if msgs == 1 else 's'} "
                         f"affected</div></div>")
        st.write("")
        actions = st.multiselect("Show actions", shown, default=shown, label_visibility="collapsed")
        view = recs[recs["action"].isin(actions)]
        st.dataframe(
            view[["action", "item", "messages", "topic", "evidence"]], hide_index=True, use_container_width=True,
            column_config={
                "action": st.column_config.TextColumn("Action", width="medium"),
                "item": st.column_config.TextColumn("What", width="large"),
                "messages": st.column_config.NumberColumn("Messages", help="Customer messages affected"),
                "topic": st.column_config.TextColumn("Topic"),
                "evidence": st.column_config.TextColumn("Why", width="large"),
            },
        )
        st.download_button("Download actions (CSV)", view.to_csv(index=False).encode("utf-8"),
                           file_name="nova_kb_actions.csv", mime="text/csv", icon=":material/download:")

    # ---------------------------------------------------------------- refund agent
    refunds = I.refund_conversations(df)
    if len(refunds):
        st.html("<div class='ni-h'>Refund agent outcomes</div>"
                "<div class='ni-sub'>Final outcome of each refund conversation</div>")
        r1, r2 = st.columns([1.35, 1])
        rdf = refunds.rename_axis("outcome").reset_index(name="conversations")
        chart = alt.Chart(rdf).mark_bar(size=18, color=SERIES_BLUE, cornerRadiusEnd=4).encode(
            y=alt.Y("outcome:N", sort="-x", title=None,
                    axis=alt.Axis(grid=False, labelLimit=200, labelOverlap=False, labelPadding=8)),
            x=alt.X("conversations:Q", title="Conversations", axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("outcome:N", title="Outcome"), alt.Tooltip("conversations:Q", title="Conversations")],
        )
        labels = alt.Chart(rdf).mark_text(align="left", dx=6, color=INK, fontSize=12).encode(
            y=alt.Y("outcome:N", sort="-x"), x="conversations:Q", text="conversations:Q")
        r1.altair_chart(_axis_cfg((chart + labels).properties(height=32 * len(rdf) + 30)), use_container_width=True)
        reasons = I.rejection_reasons(df)
        with r2:
            st.html("<div class='ni-sub' style='margin-top:6px'><b>Why refunds were refused</b></div>")
            if len(reasons):
                st.dataframe(reasons.rename_axis("reason").reset_index(name="times"), hide_index=True,
                             use_container_width=True)
            else:
                st.caption("No refused refunds in this period.")

    with st.expander("Raw conversation log", icon=":material/list_alt:"):
        raw = df[["ts", "channel", "route", "topic", "question", "fallback", "rating", "best_ref", "best_score"]]
        st.dataframe(raw.sort_values("ts", ascending=False), hide_index=True, use_container_width=True)


tab_conv, tab_cost = st.tabs([":material/forum: Conversations & knowledge base", ":material/payments: Token costs"])
with tab_conv:
    if log.empty:
        st.info("No conversations logged yet. Chat with Nova, or create demo data with "
                "`.venv\\Scripts\\python scripts\\seed_demo_data.py`, then refresh.")
    elif df.empty:
        st.warning("No conversations match these filters.")
    else:
        render_conversations(df)
with tab_cost:
    import cost_view

    cost_view.render(df, period_days=days, axis_cfg=_axis_cfg)
