"""Streamlit chat UI for Nova, the NovaCell support assistant.

Run:  .venv\\Scripts\\streamlit run app.py
"""
import html

import streamlit as st

from chain import MissingApiKey, log_feedback, log_interaction, stream_answer
from config import ensure_ingested, warm_stores
from retriever import COLLECTIONS, retrieve

BLUE = "#0A3A8C"
BLUE_LIGHT = "#EEF4FC"
ORANGE = "#F26B1D"

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "I was charged twice for my monthly plan. What should I do?",
    "My eSIM won't activate. How do I fix it?",
    "How do I set up roaming before I travel?",
    "My calls keep dropping. What should I do?",
    "How do I check my current data balance?",
    "I can't log in to the NovaCell app.",
]

# (icon, title, blurb, question) for the start-screen topic cards
TOPICS = [
    (":material/network_check:", "Slow or no data", "Speeds, signal, APN", SAMPLE_QUESTIONS[0]),
    (":material/receipt_long:", "Bills & charges", "Unexpected charges", SAMPLE_QUESTIONS[1]),
    (":material/sim_card:", "SIM & eSIM", "Activation and swaps", SAMPLE_QUESTIONS[2]),
    (":material/flight_takeoff:", "Roaming", "Set up before you travel", SAMPLE_QUESTIONS[3]),
    (":material/call:", "Calls", "Drops, echo, Wi-Fi calling", SAMPLE_QUESTIONS[4]),
    (":material/account_circle:", "Account & app", "Login, balance, settings", SAMPLE_QUESTIONS[6]),
]

SOURCE_STYLE = {  # label, badge colours
    "FAQ": ("FAQ", BLUE, "#FFFFFF"),
    "TICKETS": ("Ticket", ORANGE, "#FFFFFF"),
    "GUIDES": ("Guide", "#DCE7F7", BLUE),
}

AVATARS = {"user": ":material/person:", "assistant": ":material/support_agent:"}

st.set_page_config(page_title="NovaCell Support", page_icon="📶", layout="centered")

st.html(
    f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 820px; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  /* hero */
  .nc-hero {{
    background: linear-gradient(120deg, {BLUE} 0%, #1556C0 100%);
    border-radius: 18px; padding: 26px 28px 22px; color: #fff;
    border-bottom: 5px solid {ORANGE}; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(10, 58, 140, .18);
  }}
  .nc-logo {{ font-weight: 800; font-size: 1.05rem; letter-spacing: .02em; opacity: .95; }}
  .nc-logo b {{ color: {ORANGE}; }}
  .nc-hero h1 {{ color: #fff; font-size: 2rem; margin: 6px 0 4px; padding: 0; line-height: 1.2; }}
  .nc-hero p {{ margin: 0; opacity: .88; font-size: .98rem; }}
  .nc-pill {{
    display: inline-block; margin-top: 14px; padding: 4px 12px; border-radius: 999px;
    background: rgba(255,255,255,.14); font-size: .8rem;
  }}
  .nc-pill i {{ display:inline-block; width:8px; height:8px; border-radius:50%;
               background:{ORANGE}; margin-right:6px; font-style:normal; }}

  /* topic cards */
  .st-key-topics .stButton button {{
    height: 100%; min-height: 92px; width: 100%; text-align: left; justify-content: flex-start;
    background: #fff; border: 1.5px solid #D6E2F3; border-radius: 14px; color: {BLUE};
    padding: 14px 16px; transition: all .15s ease;
  }}
  .st-key-topics .stButton button:hover {{
    border-color: {ORANGE}; box-shadow: 0 6px 16px rgba(242,107,29,.15); transform: translateY(-1px);
    color: {BLUE};
  }}
  .st-key-topics .stButton button > div {{ justify-content: flex-start; text-align: left; }}
  .st-key-topics .stButton button p {{ font-size: .9rem; }}
  @media (max-width: 640px) {{
    .st-key-topics .stButton button {{ min-height: 0; padding: 10px 14px; }}
    .st-key-topics [data-testid="stHorizontalBlock"] {{ gap: .5rem; }}
    .nc-hero {{ padding: 20px 18px 16px; }}
    .nc-hero h1 {{ font-size: 1.5rem; }}
  }}
  .st-key-topics .stButton button span[data-testid="stIconMaterial"] {{ color: {ORANGE}; font-size: 1.6rem; }}
  .nc-section {{ color: {BLUE}; font-weight: 700; font-size: .95rem; margin: 6px 0 2px; }}

  /* chat bubbles */
  [data-testid="stChatMessage"] {{
    border-radius: 16px; padding: 14px 16px; margin-bottom: 10px;
    background: #fff; border: 1px solid #E3EAF5;
  }}
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
  [data-testid="stChatMessage"]:has(.nc-user) {{
    background: {BLUE_LIGHT}; border-color: {BLUE_LIGHT};
  }}
  [data-testid="stChatMessage"]:has(.nc-bot) {{ border-left: 4px solid {ORANGE}; }}
  [data-testid="stChatMessageAvatarCustom"], [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessageAvatarAssistant"] {{ background: {BLUE}; color: #fff; }}
  [data-testid="stChatMessage"]:has(.nc-bot) [data-testid^="stChatMessageAvatar"] {{ background: {ORANGE}; }}

  /* source badges */
  .nc-src {{ display:flex; align-items:center; gap:10px; padding:6px 0; border-bottom:1px dashed #E3EAF5; }}
  .nc-src:last-child {{ border-bottom: 0; }}
  .nc-badge {{ font-size:.72rem; font-weight:700; padding:2px 9px; border-radius:999px; white-space:nowrap; }}
  .nc-ref {{ font-family: ui-monospace, Consolas, monospace; font-size:.8rem; color:{BLUE}; white-space:nowrap; }}
  .nc-title {{ font-size:.85rem; color:#33475F; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .nc-score {{ font-size:.75rem; color:#7A8BA3; }}

  /* chat input */
  [data-testid="stChatInput"] {{ border: 1.5px solid #C9D8EE; border-radius: 14px; }}
  [data-testid="stChatInput"]:focus-within {{ border-color: {ORANGE}; }}
  [data-testid="stChatInputSubmitButton"] {{ background: {ORANGE}; color: #fff; border-radius: 10px; }}

  /* sidebar */
  section[data-testid="stSidebar"] .stButton button {{
    background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.22); color: #fff;
    text-align: left; justify-content: flex-start; border-radius: 10px;
  }}
  section[data-testid="stSidebar"] .stButton button:hover {{ border-color: {ORANGE}; background: rgba(242,107,29,.18); color:#fff; }}
  section[data-testid="stSidebar"] .st-key-clear button {{ background: {ORANGE}; border-color: {ORANGE}; justify-content: center; font-weight: 600; }}
  section[data-testid="stSidebar"] .st-key-clear button:hover {{ background: #D95A10; }}
  .nc-side-logo {{ font-size: 1.6rem; font-weight: 800; color: #fff; letter-spacing: .01em; }}
  .nc-side-logo b {{ color: {ORANGE}; }}
  .nc-side-sub {{ color: rgba(255,255,255,.75); font-size: .85rem; margin-top: -4px; }}
  .nc-side-h {{ color: #fff; font-weight: 700; font-size: .8rem; text-transform: uppercase;
               letter-spacing: .08em; opacity: .8; margin: 18px 0 6px; }}
  .nc-card {{ background: rgba(255,255,255,.1); border-left: 3px solid {ORANGE}; border-radius: 8px;
             padding: 10px 12px; color: #fff; font-size: .85rem; line-height: 1.45; }}
  .nc-foot {{ text-align:center; color:#7A8BA3; font-size:.78rem; margin-top: 8px; }}
</style>
"""
)


@st.cache_resource(show_spinner="Loading the NovaCell knowledge base…")
def warm_up():
    ensure_ingested()  # first start on a fresh deploy builds chroma_store/ from data/
    warm_stores(COLLECTIONS)
    return True


warm_up()

if "messages" not in st.session_state:
    st.session_state.messages = []


def ask(question: str) -> None:
    st.session_state.pending = question


def render_sources(docs_meta: list[dict]) -> None:
    with st.expander(f"Sources ({len(docs_meta)})", icon=":material/fact_check:"):
        rows = []
        for m in docs_meta:
            label, bg, fg = SOURCE_STYLE.get(m["collection"], (m["collection"], BLUE, "#fff"))
            rows.append(
                f"<div class='nc-src'><span class='nc-badge' style='background:{bg};color:{fg}'>{label}</span>"
                f"<span class='nc-ref'>{html.escape(str(m['ref']))}</span>"
                f"<span class='nc-title'>{html.escape(str(m['title']))}</span>"
                f"<span class='nc-score'>{m['score']:.2f}</span></div>"
            )
        st.html("".join(rows))


def render_feedback(msg: dict) -> None:
    key = f"fb_{msg['id']}"

    def on_change():
        value = st.session_state.get(key)
        if value is not None:
            log_feedback(msg["id"], "up" if value == 1 else "down")

    st.feedback("thumbs", key=key, on_change=on_change)


def role_marker(role: str) -> None:
    """Invisible marker so CSS can style user vs assistant bubbles."""
    st.html(f"<span class='{'nc-user' if role == 'user' else 'nc-bot'}'></span>")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.html(
        "<div class='nc-side-logo'>Nova<b>Cell</b></div>"
        "<div class='nc-side-sub'>Customer Support</div>"
        "<div class='nc-side-h'>Popular questions</div>"
    )
    for q in SAMPLE_QUESTIONS:
        st.button(q, use_container_width=True, on_click=ask, args=(q,))
    st.html(
        "<div class='nc-side-h'>Good to know</div>"
        "<div class='nc-card'>Nova answers only from NovaCell's FAQ, resolved support cases and "
        "technical guides. It can't see your account, balance or bill. For account help call "
        "<b>611</b> or use the NovaCell app.</div>"
    )
    st.write("")
    if st.button("New conversation", key="clear", icon=":material/refresh:", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending", None)
        st.rerun()

# ---------------------------------------------------------------- header
typed = st.chat_input("Ask Nova about data, bills, SIM, roaming or calls…")
question = typed or st.session_state.pop("pending", None)

st.html(
    "<div class='nc-hero'>"
    "<div class='nc-logo'>Nova<b>Cell</b> &nbsp;·&nbsp; Support</div>"
    "<h1>Hi, I'm Nova. How can I help?</h1>"
    "<p>Quick answers to everyday mobile questions, straight from NovaCell's verified help content.</p>"
    "<span class='nc-pill'><i></i>Available 24/7 &nbsp;·&nbsp; Every answer shows its sources</span>"
    "</div>"
)

if not st.session_state.messages and not question:
    st.html("<div class='nc-section'>Choose a topic to get started</div>")
    with st.container(key="topics"):
        for row in (TOPICS[:3], TOPICS[3:]):
            for col, (icon, title, blurb, q) in zip(st.columns(3), row):
                col.button(f"**{title}**  \n{blurb}", key=f"topic_{title}", icon=icon,
                           use_container_width=True, on_click=ask, args=(q,))

# ---------------------------------------------------------------- history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        role_marker(msg["role"])
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])
            render_feedback(msg)

# ---------------------------------------------------------------- new turn
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        role_marker("user")
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        role_marker("assistant")
        with st.spinner("Nova is checking NovaCell's knowledge base…"):
            docs = retrieve(question)
        try:
            text = st.write_stream(stream_answer(question, docs))
        except MissingApiKey as exc:
            st.error(str(exc), icon=":material/key:")
            st.session_state.messages.pop()
            st.stop()
        except Exception as exc:  # network / API errors
            if type(exc).__name__ == "RateLimitError":
                st.warning("Nova is busy right now. Please try again in a minute.", icon=":material/hourglass_top:")
            else:
                st.error(f"Sorry, Nova is unavailable right now ({type(exc).__name__}). Please call 611.")
            st.session_state.messages.pop()
            st.stop()

        sources = [
            {k: d.metadata[k] for k in ("collection", "ref", "title", "score")} for d in docs
        ]
        interaction_id = log_interaction(question, text, docs, channel="web")
        msg = {"role": "assistant", "content": text, "sources": sources, "id": interaction_id}
        st.session_state.messages.append(msg)
        render_sources(sources)
        render_feedback(msg)

if st.session_state.messages:
    st.html("<div class='nc-foot'>Nova answers from NovaCell's verified help content. "
            "For anything about your own account, call 611.</div>")
