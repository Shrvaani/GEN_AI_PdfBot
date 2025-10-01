import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import uuid
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from huggingface_hub import InferenceClient

# Helper to safely extract assistant text from HF chat response
def _extract_assistant_text(chat_resp) -> str:
    try:
        choice = chat_resp.choices[0]
        msg = choice.message
        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("reasoning_content")
        else:
            content = getattr(msg, "content", None) or getattr(msg, "reasoning_content", None)
        if content and isinstance(content, str):
            return content.strip()
    except Exception:
        pass
    try:
        return str(chat_resp).strip()
    except Exception:
        return ""

# --- Streamlit Setup ---
load_dotenv()
st.set_page_config(page_title="RAG Chatbot (gpt-oss-20b · HF Inference API)", layout="wide")
st.title("🦉 GPT-OSS-20B Chat")

# --- Persistence helpers (JSON on local disk) ---
PERSIST_DIR = Path.home() / ".rag_pdfbot"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_FILE = PERSIST_DIR / "chats.json"

def load_persisted_state() -> dict:
    """Load chats from disk.
    Do not rely on any in-session flags; if the file exists, read it.
    """
    try:
        if PERSIST_FILE.exists():
            data = json.loads(PERSIST_FILE.read_text())
            if isinstance(data, dict) and data.get("version") == 1:
                return data
    except Exception as e:
        st.warning(f"Failed to load persisted state: {str(e)}")
    return {"version": 1, "chats": {}, "active_chat_id": None}

def save_persisted_state(chats: dict, active_chat_id: str | None) -> bool:
    """Persist chats in a JSON-safe structure.
    - Converts message tuples and Document objects into plain JSON.
    - Retries a few times before failing.
    """
    def _to_safe_messages(msgs):
        safe = []
        for m in msgs or []:
            if isinstance(m, dict):
                q = m.get("q") or m.get("question") or (m.get("role") == "user" and m.get("content"))
                a = m.get("a") or m.get("answer") or (m.get("role") == "assistant" and m.get("content"))
                sources = []
                for s in m.get("sources", []) or []:
                    if isinstance(s, dict):
                        sources.append({
                            "label": s.get("label"),
                            "page": s.get("page"),
                            "content": s.get("content"),
                        })
                    else:
                        content = getattr(s, "page_content", None)
                        meta = getattr(s, "metadata", {})
                        page = meta.get("page") if isinstance(meta, dict) else None
                        sources.append({"page": page, "content": content})
                safe.append({"q": q, "a": a, "sources": sources})
            elif isinstance(m, (list, tuple)) and len(m) >= 2:
                q, a = m[0], m[1]
                sources = []
                if len(m) >= 3:
                    for s in m[2] or []:
                        content = getattr(s, "page_content", None)
                        meta = getattr(s, "metadata", {})
                        page = meta.get("page") if isinstance(meta, dict) else None
                        sources.append({"page": page, "content": content})
                safe.append({"q": q, "a": a, "sources": sources})
            else:
                safe.append({"q": None, "a": None, "sources": []})
        return safe

    # Build JSON-safe chat payload
    safe_chats = {}
    for cid, chat in chats.items():
        title = chat.get("title", "New Chat") if isinstance(chat, dict) else "New Chat"
        msgs = chat.get("messages", []) if isinstance(chat, dict) else []
        safe_chats[cid] = {"title": title, "messages": _to_safe_messages(msgs)}

    for attempt in range(3):
        try:
            payload = {"version": 1, "chats": safe_chats, "active_chat_id": active_chat_id}
            PERSIST_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            return True
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed to save chat state: {str(e)}")
            import time
            time.sleep(0.5 * (attempt + 1))
    return False

# Consolidated CSS (unchanged)
st.markdown(
    """
    <style>
    :root {
        --brand-blue: #6366f1;
        --brand-purple: #a855f7;
        --sidebar-bg: #f0f4ff;
        --main-bg: #ffffff;
        --button-gradient: linear-gradient(to right, var(--brand-blue), var(--brand-purple));
        --title-gradient: linear-gradient(to right, #6366f1, #a855f7);
        --radius: 8px;
    }
    .block-container { max-width: 860px; padding-top: 4rem; }
    html, body, [data-testid="stAppViewContainer"] { font-size: 15px; }
    hr { display: none !important; }
    section[data-testid="stSidebar"] { background: var(--sidebar-bg) !important; }
    section[data-testid="stSidebar"] .block-container { background: transparent; padding: 14px; margin: 0; }
    section[data-testid="stSidebar"] h2 { font-weight: 700; margin-bottom: 0.25rem; }
    section[data-testid="stSidebar"] .stMarkdown { opacity: 0.95; }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        border-radius: var(--radius); border: 1px solid #e0e7ff; background: white;
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: var(--radius); font-weight: 600; transition: all 0.2s ease;
        background: var(--button-gradient); color: white !important; border: none;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { filter: brightness(1.1); }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: var(--button-gradient); opacity: 0.8;
    }
    div[data-testid="stAppViewContainer"] > .main { background: var(--main-bg); }
    h1 {
        background: var(--title-gradient); color: white; padding: 12px 20px;
        border-radius: var(--radius); text-align: center; font-size: 1.8rem !important;
    }
    h1 + div[data-testid="stMarkdownContainer"] p {
        text-align: center; color: #6b7280; margin-top: -10px; font-size: 0.95rem;
    }
    div[data-testid="stFileUploader"] > section {
        background: white; border: 1px solid #e0e7ff; border-radius: var(--radius); padding: 10px 12px;
    }
    .stChatMessage div[data-testid="stMarkdownContainer"] { font-size: 0.98rem; }
    .stChatMessage[aria-label="assistant"] div[data-testid="stMarkdownContainer"] {
        background: white; border: 1px solid #e0e7ff; border-left: 4px solid var(--brand-blue);
        border-radius: var(--radius); padding: 12px;
    }
    .stChatMessage[aria-label="user"] div[data-testid="stMarkdownContainer"] {
        background: #f9fafb; border: 1px solid #e0e7ff; border-right: 4px solid var(--brand-purple);
        border-radius: var(--radius); padding: 12px;
    }
    div[data-testid="stChatInput"] textarea {
        min-height: 42px !important; font-size: 0.98rem; border-radius: var(--radius);
        background: #f0f4ff; border: 1px solid #c7d2fe;
    }
    div[data-baseweb="notification"] { border-radius: var(--radius); }
    div.stExpander { border-radius: var(--radius); border: 1px solid #e0e7ff; }
    section[data-testid="stSidebar"] .stButton { margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Multi-chat state ---
# Always hydrate from disk once per session start
if "hydrated" not in st.session_state:
    persisted = load_persisted_state()
    st.session_state.chats = persisted.get("chats", {}) or {}
    st.session_state.active_chat_id = persisted.get("active_chat_id")
    st.session_state.hydrated = True
    # If nothing on disk, initialize a default chat
    if not st.session_state.chats:
        _id = str(uuid.uuid4())
        st.session_state.chats[_id] = {"title": "Chat 1", "messages": []}
        st.session_state.active_chat_id = _id
        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)

def _create_new_chat(title: str | None = None):
    chat_id = str(uuid.uuid4())  # Use UUID for unique chat ID
    if not title:
        title = f"Chat {len(st.session_state.chats)+1}"
    st.session_state.chats[chat_id] = {"title": title, "messages": []}
    st.session_state.active_chat_id = chat_id
    if not save_persisted_state(st.session_state.chats, st.session_state.active_chat_id):
        st.error("Failed to create new chat due to persistence error.")
    else:
        st.success("New chat created!")

def _get_active_chat():
    cid = st.session_state.active_chat_id
    if cid and cid in st.session_state.chats:
        return st.session_state.chats[cid]
    if not st.session_state.chats:
        _create_new_chat("Chat 1")
    else:
        st.session_state.active_chat_id = next(iter(st.session_state.chats))
    return st.session_state.chats[st.session_state.active_chat_id]

# Sidebar: chat manager
with st.sidebar:
    st.header("GPT-OSS-20B\nChat")
    if "reasoning_level" not in st.session_state:
        st.session_state.reasoning_level = "Medium"
    st.caption("Reasoning Level")
    st.session_state.reasoning_level = st.selectbox(
        "Reasoning Level",
        options=["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(st.session_state.reasoning_level),
        label_visibility="collapsed",
        key="reasoning_level_select"
    )
    if st.button("New Chat", key="new_chat", use_container_width=True):
        _create_new_chat(f"Chat {len(st.session_state.chats)+1}")
        st.rerun()

    st.subheader("Conversations")
    if st.button("Clear Current Chat", key="clear_chat", use_container_width=True):
        active_chat = _get_active_chat()
        active_chat["messages"] = []
        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
        st.rerun()

    # Inline rename state
    if "rename_id" not in st.session_state:
        st.session_state.rename_id = None
        st.session_state.rename_value = ""

    # List chats with robust rename/delete behavior
    for cid, data in list(st.session_state.chats.items()):
        title = data.get("title") or f"Chat {cid[:8]}"
        btn_type = "primary" if cid == st.session_state.active_chat_id else "secondary"
        if st.button(title, key=f"chat_btn_{cid}", use_container_width=True, type=btn_type):
            st.session_state.active_chat_id = cid
            save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Rename", key=f"ren_{cid}", use_container_width=True, type="secondary"):
                st.session_state.rename_id = cid
                st.session_state.rename_value = title
                st.rerun()
        with c2:
            if st.button("Delete", key=f"del_{cid}", use_container_width=True, type="secondary"):
                if cid in st.session_state.chats:
                    was_active = (st.session_state.active_chat_id == cid)
                    st.session_state.chats.pop(cid, None)
                    if was_active:
                        st.session_state.active_chat_id = next(iter(st.session_state.chats), None)
                        # Also clear any loaded PDF/index tied to the deleted chat
                        st.session_state.index = []
                        st.session_state.corpus_id = None
                    save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
                    st.rerun()

        # Inline rename UI
        if st.session_state.rename_id == cid:
            new_title = st.text_input(
                "Rename chat",
                value=st.session_state.rename_value,
                key=f"ren_input_{cid}",
                label_visibility="collapsed",
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Save", key=f"ren_save_{cid}", use_container_width=True, type="secondary"):
                    if new_title.strip():
                        st.session_state.chats[cid]["title"] = new_title.strip()
                        st.session_state.rename_id = None
                        st.session_state.rename_value = ""
                        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)
                        st.rerun()
                    else:
                        st.warning("Title cannot be empty")
            with rc2:
                if st.button("Cancel", key=f"ren_cancel_{cid}", use_container_width=True, type="secondary"):
                    st.session_state.rename_id = None
                    st.session_state.rename_value = ""
                    st.rerun()

# --- Upload PDF and rest of the code (unchanged) ---
uploaded_files = st.file_uploader("📤 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    corpus_id = ";".join([f"{f.name}:{getattr(f, 'size', 0)}" for f in uploaded_files])
    if st.session_state.get("corpus_id") != corpus_id:
        st.session_state.corpus_id = corpus_id
        st.session_state.index = []
        for chat in st.session_state.chats.values():
            chat["messages"] = []
        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)

    with st.spinner("⏳ Processing PDF..."):
        docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for uploaded in uploaded_files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
            tmp_path = tmp.name
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load()
            docs.extend(splitter.split_documents(pages))

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            st.error("Missing HF_TOKEN environment variable. Create a token at https://huggingface.co/settings/tokens and set HF_TOKEN.")
            st.stop()
        embed_client = InferenceClient(model="sentence-transformers/all-mpnet-base-v2", token=hf_token)

        def embed_texts(text_list: list[str]) -> list[np.ndarray]:
            if not text_list:
                return []
            out: list[np.ndarray] = []
            batch_size_local = 12
            for j in range(0, len(text_list), batch_size_local):
                sub = text_list[j:j+batch_size_local]
                vecs = None
                for attempt in range(3):
                    try:
                        vecs = embed_client.feature_extraction(sub)
                        break
                    except Exception:
                        import time
                        time.sleep(1.2 * (attempt + 1))
                if vecs is None:
                    raise RuntimeError("Embedding request failed repeatedly.")
                if isinstance(vecs, list) and vecs and isinstance(vecs[0], list):
                    out.extend([np.array(v, dtype=float) for v in vecs])
                elif isinstance(vecs, list) and vecs and isinstance(vecs[0], (int, float)):
                    v = np.array(vecs, dtype=float)
                    out.extend([v for _ in range(len(sub))])
                else:
                    arr = np.array(vecs, dtype=float)
                    if arr.ndim == 1:
                        out.extend([arr for _ in range(len(sub))])
                    elif arr.ndim == 2 and arr.shape[0] == len(sub):
                        out.extend([arr[k] for k in range(arr.shape[0])])
                    else:
                        first = arr[0] if arr.ndim > 1 else arr
                        out.extend([first for _ in range(len(sub))])
            return out

        texts = [d.page_content for d in docs]
        embeddings = embed_texts(texts)
        n = min(len(embeddings), len(docs))
        embeddings = embeddings[:n]
        docs = docs[:n]
        st.session_state.index = [
            {"embedding": embeddings[i], "doc": docs[i]} for i in range(n)
        ]
        client = InferenceClient(model="openai/gpt-oss-20b", token=hf_token)

    st.success("✅ PDF processed. Ask your questions!")

    active_chat = _get_active_chat()
    for entry in active_chat["messages"]:
        q = entry[0]
        a = entry[1]
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
            if len(entry) > 2 and entry[2]:
                with st.expander("Sources"):
                    for i, d in enumerate(entry[2]):
                        src_label = f"Source {i+1}"
                        page_info = d.metadata.get("page") if isinstance(d.metadata, dict) else None
                        page_str = f" (page {page_info})" if page_info is not None else ""
                        st.write(f"- {src_label}{page_str}")

    query = st.chat_input("Type your message here...", key=f"query_{st.session_state.active_chat_id}")

    if query:
        active_chat = _get_active_chat()
        insertion_idx = len(active_chat["messages"])
        active_chat["messages"].append((query, "", []))
        save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("🤖 Generating answer..."):
            def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                return float(np.dot(a, b) / denom)

            q_vec = embed_client.feature_extraction(query)
            if isinstance(q_vec, list) and isinstance(q_vec[0], list):
                q_vec = np.array(q_vec[0], dtype=float)
            else:
                q_vec = np.array(q_vec, dtype=float)

            scored = []
            for item in st.session_state.index:
                score = cosine_similarity(q_vec, item["embedding"])
                scored.append((score, item["doc"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            top_k = 12
            SIM_THRESHOLD = 0.15
            filtered = [(s, d) for s, d in scored if s >= SIM_THRESHOLD][:top_k]
            if not filtered:
                filtered = scored[:3]
            source_docs = [d for _, d in filtered]
            context_blocks = []
            for idx, d in enumerate(source_docs):
                label = f"[Source {idx+1}]"
                content = d.page_content
                context_blocks.append(f"{label}\n{content}")
            context_text = "\n\n".join(context_blocks) if context_blocks else "(No context retrieved)"

            system = (
                "You are a helpful RAG assistant. Answer the user's question using only the provided context. "
                "If the answer is not present in the context, say you don't know. "
                "Cite sources inline like [Source 1], [Source 2]."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}" if source_docs else f"No relevant context found. Question: {query}"},
            ]

            try:
                chat_resp = client.chat_completion(
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.2,
                    stream=False,
                )
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Generation failed: {str(e)}")
                raise

            answer = _extract_assistant_text(chat_resp)
            if not answer or not isinstance(answer, str) or not answer.strip():
                # Fallback: try plain text_generation on the constructed prompt
                try:
                    tg_client = InferenceClient("openai/gpt-oss-20b", token=hf_token)
                    gen = tg_client.text_generation(
                        prompt,
                        max_new_tokens=2048,
                        temperature=0.2,
                        do_sample=False,
                    )
                    if isinstance(gen, str) and gen.strip():
                        answer = gen.strip()
                except Exception:
                    pass
            with st.chat_message("assistant"):
                st.markdown(answer)
            active_chat["messages"][insertion_idx] = (query, answer, source_docs)
            save_persisted_state(st.session_state.chats, st.session_state.active_chat_id)

else:
    st.info("📥 Please upload a PDF to begin.")
    idle_query = st.chat_input("Type your message here...")
    if idle_query:
        st.warning("Upload a PDF first so I can ground my answers in your document.")