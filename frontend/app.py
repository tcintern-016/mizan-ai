"""
Pakistan Law Assistant — Streamlit Frontend.
Calls the FastAPI backend over HTTP (works both locally and against a
deployed backend — just set API_BASE_URL).
"""

import sys
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import time
from datetime import datetime
from typing import Optional

import streamlit as st
import requests

st.set_page_config(
    page_title="Pakistan Law Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pull Streamlit Cloud secrets (if any) into os.environ so a plain
# os.environ.get also works when running locally with a .env file.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # st.secrets raises if no secrets.toml exists locally — fine

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

DOCUMENT_OPTIONS = [
    "All Documents",
    "Constitution of Pakistan",
    "Pakistan Penal Code",
    "Contract Act 1872",
    "PECA 2016 (Prevention of Electronic Crimes Act)",
]


# ── CSS ────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
<style>
:root {
    --navy: #0B1220; --navy2: #0D1B2A; --emerald: #14532D;
    --emerald2: #166534; --gold: #B08D57; --white: #F0F4FF;
    --muted: #8899BB; --card: #101B2D; --border: #22314A; --radius: 14px;
}
html, body, [class*="css"] { font-family: 'Georgia', serif; background-color: var(--navy) !important; color: var(--white) !important; }
.stApp { background-color: var(--navy) !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { background: var(--navy2) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--white) !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--white) !important; padding: 12px 16px !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--emerald) 0%, var(--emerald2) 100%) !important;
    color: white !important; border: none !important; border-radius: var(--radius) !important;
    font-weight: 600 !important; padding: 10px 24px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


def hero_header():
    st.markdown(
        """
<div style="text-align:center; padding: 24px 0;">
  <div style="font-size:44px;">⚖️</div>
  <h1 style="font-size:2.4rem; font-weight:700; color:#B08D57; margin:8px 0;">Pakistan Law Assistant</h1>
  <p style="color:#8899BB; font-size:0.95rem;">CONSTITUTION · PENAL CODE · CONTRACT ACT · PECA</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ── API helpers ──────────────────────────────────────────────────────────────
def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_ask(question: str, top_k: int, source_filter: Optional[str]):
    try:
        payload = {"question": question, "top_k": top_k, "source_filter": source_filter}
        r = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to backend at {API_BASE}. Is it running / deployed?")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_search(query: str, top_k: int, source_filter: Optional[str]):
    try:
        payload = {"query": query, "top_k": top_k, "source_filter": source_filter}
        r = requests.post(f"{API_BASE}/search", json=payload, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Session state ────────────────────────────────────────────────────────────
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "health" not in st.session_state:
        st.session_state.health = None
    if "last_health_check" not in st.session_state:
        st.session_state.last_health_check = 0


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#B08D57;'>⚖️ Pakistan Law Assistant</h2>",
            unsafe_allow_html=True,
        )
        top_k = st.slider("Sources to retrieve", 1, 15, 5)
        source_choice = st.selectbox("Restrict to document", DOCUMENT_OPTIONS)
        source_filter = None if source_choice == "All Documents" else source_choice

        st.markdown("---")
        now = time.time()
        if now - st.session_state.last_health_check > 30:
            st.session_state.health = api_health()
            st.session_state.last_health_check = now

        health = st.session_state.health
        if health:
            st.markdown("**System Status**")
            st.write(f"API: {health.get('status', 'unknown').upper()}")
            st.write(f"Groq configured: {health.get('groq_configured')}")
            st.write(f"Chunks indexed: {health.get('chunk_count', 0):,}")
            sources = health.get("available_sources", [])
            if sources:
                st.write("Documents loaded:")
                for s in sources:
                    st.write(f"- {s}")
        else:
            st.error("Backend offline. Check API_BASE_URL / deployment.")

        if st.session_state.messages:
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            export_data = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Export Chat",
                data=export_data,
                file_name=f"law_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )

    return top_k, source_filter


# ── Chat page ────────────────────────────────────────────────────────────────
def render_chat_page(top_k: int, source_filter: Optional[str]):
    hero_header()

    if not st.session_state.messages:
        suggestions = [
            "What does the Constitution say about the right to freedom of speech?",
            "What is the punishment for theft under the Pakistan Penal Code?",
            "What makes a contract valid under the Contract Act 1872?",
            "What does PECA say about unauthorized access to a computer system?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state.pending_question = s
                    st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} source(s) cited"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['citation']}** — {s['confidence_label']} ({s['confidence']}%)"
                        )
                        st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))

    pending = st.session_state.pop("pending_question", None)
    question = st.chat_input("Ask about Pakistani law…") or pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching legal documents…"):
                result = api_ask(question, top_k, source_filter)
            if result:
                st.markdown(result["answer"])
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📚 {len(sources)} source(s) cited"):
                        for s in sources:
                            st.markdown(
                                f"**{s['citation']}** — {s['confidence_label']} ({s['confidence']}%)"
                            )
                            st.caption(s["text"][:300] + ("…" if len(s["text"]) > 300 else ""))
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": sources}
                )


# ── Search page ──────────────────────────────────────────────────────────────
def render_search_page(top_k: int, source_filter: Optional[str]):
    st.markdown("## 🔍 Semantic Search")
    st.caption("Search directly across the indexed legal documents — no LLM involved.")

    query = st.text_input("Search query", placeholder="e.g. freedom of speech, theft, contract")
    if st.button("Search") and query.strip():
        with st.spinner("Searching…"):
            result = api_search(query.strip(), top_k, source_filter)
        if result and result.get("results"):
            st.write(f"**{len(result['results'])} results for \"{query}\"**")
            for r in result["results"]:
                with st.container(border=True):
                    st.markdown(f"**{r['citation']}** — {r['confidence_label']} ({r['confidence']}%)")
                    st.write(r["text"])
        elif result:
            st.info("No results found. Try a different search term.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    init_session()
    top_k, source_filter = render_sidebar()

    tab1, tab2 = st.tabs(["💬 Chat", "🔍 Search"])
    with tab1:
        render_chat_page(top_k, source_filter)
    with tab2:
        render_search_page(top_k, source_filter)


if __name__ == "__main__":
    main()
