import os
import warnings
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*sparse_softmax.*')
warnings.filterwarnings('ignore', message='.*reset_default_graph.*')

import streamlit as st
from backend.rag_pipeline import run_rag

st.set_page_config(
    page_title="Multi-Intent Technical Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    * { box-sizing: border-box; }
    body, .stApp { font-family: 'Inter', sans-serif; background: #f0f2f5; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: #e63946; color: white; border: none;
        border-radius: 8px; padding: 0.5rem 1.2rem;
        font-weight: 600; width: 100%; transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover { background: #c1121f; }

    /* Hide default streamlit chat message chrome */
    [data-testid="stChatMessage"] { background: transparent !important; padding: 0 !important; }
    [data-testid="stChatMessage"] > div { background: transparent !important; }

    /* User bubble */
    .user-row {
        display: flex; justify-content: flex-end;
        margin: 0.4rem 0;
    }
    .user-bubble {
        background: #0071e3; color: #fff;
        padding: 0.75rem 1.1rem;
        border-radius: 20px 20px 4px 20px;
        max-width: 68%; font-size: 0.93rem; line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,113,227,0.25);
        word-wrap: break-word;
    }

    /* Assistant bubble */
    .assistant-row {
        display: flex; justify-content: flex-start;
        align-items: flex-start; gap: 0.6rem;
        margin: 0.4rem 0;
    }
    .assistant-avatar {
        width: 34px; height: 34px; border-radius: 50%;
        background: #1a1a2e; display: flex; align-items: center;
        justify-content: center; font-size: 1rem;
        flex-shrink: 0; margin-top: 2px;
    }
    .assistant-bubble {
        background: #ffffff; color: #1c1c1e;
        padding: 0.9rem 1.1rem;
        border-radius: 20px 20px 20px 4px;
        max-width: 72%; font-size: 0.93rem; line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 3px solid #0071e3;
        word-wrap: break-word;
    }
    .assistant-bubble h2,
    .assistant-bubble h3,
    .assistant-bubble h4 { margin: 0.6em 0 0.25em; color: #1a1a2e; }
    .assistant-bubble ul  { margin: 0.3em 0 0.3em 1.2em; padding: 0; }
    .assistant-bubble li  { margin: 0.2em 0; }
    .assistant-bubble p   { margin: 0.3em 0; }

    /* Sources */
    .sources-block {
        margin-top: 0.75rem; padding-top: 0.65rem;
        border-top: 1px solid #e8e8e8;
    }
    .sources-title {
        font-size: 0.78rem; font-weight: 600;
        color: #0071e3; margin-bottom: 0.35rem;
    }
    .source-chip {
        display: inline-block; background: #f0f6ff;
        color: #0071e3; border: 1px solid #cce0ff;
        border-radius: 20px; padding: 0.18rem 0.65rem;
        font-size: 0.75rem; margin: 0.2rem 0.2rem 0 0; font-weight: 500;
    }

    /* Typing dots */
    .typing-row {
        display: flex; align-items: center; gap: 0.6rem;
        margin: 0.4rem 0;
    }
    .typing-bubble {
        background: #fff; border-radius: 20px;
        padding: 0.7rem 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        display: flex; align-items: center; gap: 5px;
    }
    .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #0071e3; animation: bounce 1.2s infinite;
    }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
        40%            { transform: scale(1.1); opacity: 1; }
    }

    /* Empty state */
    .empty-state { text-align: center; padding: 3rem 1rem; color: #8e8e93; }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)


def strip_tags(text: str) -> str:
    """Remove every HTML tag from a string."""
    return re.sub(r'<[^>]+>', '', text).strip()


def _inline_md(text: str) -> str:
    """Convert inline markdown (bold / italic / code) to HTML."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__',     r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
    text = re.sub(r'_(.+?)_',       r'<em>\1</em>',         text)
    text = re.sub(
        r'`(.+?)`',
        r'<code style="background:#f0f4ff;padding:0.1em 0.3em;'
        r'border-radius:3px;font-size:0.88em;">\1</code>',
        text
    )
    return text


def markdown_to_html(text: str) -> str:
    """
    Convert a markdown-ish plain string to safe HTML for bubble rendering.
    Handles: h1-h3 headings, bullet lists (* or -), bold, italic, inline code.
    """
    lines, out, in_ul = text.split('\n'), [], False

    for raw in lines:
        line = raw.rstrip()
        matched = False
        for lvl, pat, tag in [
            (3, r'^###\s+(.*)', 'h4'),
            (2, r'^##\s+(.*)',  'h3'),
            (1, r'^#\s+(.*)',   'h2'),
        ]:
            m = re.match(pat, line)
            if m:
                if in_ul: out.append('</ul>'); in_ul = False
                out.append(f'<{tag}>{_inline_md(m.group(1))}</{tag}>')
                matched = True; break
        if matched:
            continue

        m = re.match(r'^\s*[\*\-]\s+(.*)', line)
        if m:
            if not in_ul: out.append('<ul>'); in_ul = True
            out.append(f'<li>{_inline_md(m.group(1))}</li>')
            continue

        if in_ul: out.append('</ul>'); in_ul = False

        if not line.strip():
            out.append('<br>')
            continue

        out.append(f'<p>{_inline_md(line)}</p>')

    if in_ul:
        out.append('</ul>')

    return '\n'.join(out)


_NOISE = re.compile(
    r'\b(assistant[_-]row|assistant[_-]avatar|assistant[_-]bubble|'
    r'sources[_-]block|sources[_-]title|source[_-]chip|'
    r'chat[_-]wrapper|user[_-]row|user[_-]bubble|'
    r'typing[_-]row|typing[_-]bubble)\b',
    re.IGNORECASE
)


def parse_response(raw: str) -> tuple[str, list[str]]:
    """
    Accept whatever run_rag() returns — plain markdown OR pre-built HTML —
    and return (answer_html, sources_list) with NO raw HTML leaking through.

    Strategy:
      1. Harvest source chips from HTML before stripping.
      2. If the RAG already built the assistant-bubble wrapper, pull only
         the text that lives INSIDE that wrapper (before the sources-block).
      3. Otherwise just strip all tags from the raw string.
      4. Clean noise, extract markdown-style sources, convert md → html.
    """
    sources: list[str] = []

    for chip_inner in re.findall(
        r'<span[^>]*class=["\']source-chip["\'][^>]*>(.*?)</span>',
        raw, re.DOTALL
    ):
        s = strip_tags(chip_inner).replace('📄', '').strip()
        if s:
            sources.append(s)

    bubble_body_match = re.search(
        r'<div[^>]*class=["\'][^"\']*assistant-bubble[^"\']*["\'][^>]*>'
        r'(.*?)'
        r'(?:<div[^>]*class=["\'][^"\']*sources-block[^"\']*["\']|$)',
        raw, re.DOTALL
    )
    if bubble_body_match:
        text = strip_tags(bubble_body_match.group(1))
    else:
        text = strip_tags(raw)

    text = _NOISE.sub('', text)

    if not sources:
        md_delim = "---\n📚 **Sources:**"
        inline_pat = r'\(Sources?:\s*([^)]+)\)\s*$'
        if md_delim in text:
            text, sec = text.split(md_delim, 1)
            sources = [
                s.strip().lstrip('•').strip()
                for s in sec.splitlines()
                if s.strip() and s.strip().startswith('•')
            ]
        else:
            m = re.search(inline_pat, text, re.MULTILINE)
            if m:
                sources = [s.strip() for s in m.group(1).split(',') if s.strip()]
                text = text[:m.start()].rstrip()

    text = re.sub(r'📚\s*Sources?\b', '', text)
    text = re.sub(r'^\s*[⚡📚📄]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()

    if 'not available in my knowledge base' in text.lower():
        sources = []

    return markdown_to_html(text), sources


def user_bubble_html(content: str) -> str:
    return (
        f'<div class="user-row">'
        f'<div class="user-bubble">{content}</div>'
        f'</div>'
    )


def assistant_bubble_html(content: str, sources: list[str]) -> str:
    src = ""
    if sources:
        chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
        src = (
            f'<div class="sources-block">'
            f'<div class="sources-title">📚 Sources</div>'
            f'{chips}</div>'
        )
    return (
        f'<div class="assistant-row">'
        f'<div class="assistant-avatar">⚡</div>'
        f'<div class="assistant-bubble">{content}{src}</div>'
        f'</div>'
    )


TYPING_HTML = (
    '<div class="typing-row">'
    '<div class="assistant-avatar">⚡</div>'
    '<div class="typing-bubble">'
    '<div class="dot"></div><div class="dot"></div><div class="dot"></div>'
    '</div></div>'
)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("⚡ About")
    st.write(
        "This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer "
        "questions from verified technical documents."
    )
    st.divider()
    st.subheader("📚 Knowledge Base")
    st.markdown("""
- **NEC Code** — 2023 National Electrical Code
- **Solar Manuals** — Professional installation guides
- **Wattmonk** — Company documentation
    """)
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


st.title("⚡ Multi-Intent Technical Chatbot")
st.caption("Ask questions about NEC Electrical Code, Solar Installation, or Wattmonk Company Information")


if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="icon">💬</div>'
        '<p>No messages yet.<br>'
        'Ask me anything about NEC Code, Solar Installation, or Wattmonk!</p>'
        '</div>',
        unsafe_allow_html=True
    )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(user_bubble_html(msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(
                assistant_bubble_html(msg["content"], msg.get("sources", [])),
                unsafe_allow_html=True
            )


user_input = st.chat_input("Ask a technical question…")

if user_input:
    user_text = user_input.strip()
    st.markdown(user_bubble_html(user_text), unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": user_text})

    typing_slot = st.empty()
    typing_slot.markdown(TYPING_HTML, unsafe_allow_html=True)

    try:
        raw = run_rag(user_text, chat_history=st.session_state.messages)
        answer_html, sources = parse_response(raw)
    except Exception as exc:
        answer_html = f"<p>⚠️ Error processing your question: {exc}</p>"
        sources = []

    typing_slot.markdown(
        assistant_bubble_html(answer_html, sources),
        unsafe_allow_html=True
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_html,
        "sources": sources
    })
    st.rerun()