import streamlit as st
import os, json, uuid
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
st.set_page_config(page_title="GPT-OSS-20B Chat", page_icon="🤖", layout="wide")

# Fallback: try to read HF token from a local api.txt if present (never committed)
def _fallback_read_hf_token():
    try:
        if os.path.exists("api.txt"):
            txt = open("api.txt","r",encoding="utf-8").read()
            for part in txt.replace("\n"," ").split():
                if part.startswith("hf_") and len(part) > 10:
                    return part.strip()
            for ln in txt.splitlines():
                if "HF_TOKEN" in ln and "=" in ln:
                    return ln.split("=",1)[1].strip()
    except Exception:
        pass
    return ""

CSS = """
<style>
    /* CSS Variables for theme support */
    :root {
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
        --info-box-bg: #e3f2fd;
        --info-box-border: #2196f3;
        --primary-gradient: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        --spacing-sm: 0.3rem;
        --spacing-md: 0.6rem;
        --spacing-lg: 1rem;
    }
    
    /* Streamlit light mode (default) */
    .stApp {
        --background-color: #ffffff;
        --text-color: #262730;
        --card-background: #f8f9fa;
        --border-color: #e9ecef;
        --info-box-bg: #e3f2fd;
        --info-box-border: #2196f3;
    }
    
    /* Streamlit dark mode */
    .stApp[data-theme="dark"] {
        --background-color: #0e1117;
        --text-color: #fafafa;
        --card-background: #262730;
        --border-color: #464646;
        --info-box-bg: #1e3a5f;
        --info-box-border: #4fc3f7;
    }
    
    /* App container background */
    .stApp [data-testid="stAppViewContainer"] {
        background-color: var(--background-color) !important;
        padding: var(--spacing-md) !important;
    }
    
    /* Main content area */
    .stApp .main .block-container {
        background-color: var(--background-color) !important;
        padding: var(--spacing-md) !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    
    /* Main header */
    .main-header {
        background: var(--primary-gradient) !important;
        padding: var(--spacing-lg) var(--spacing-md) !important;
        border-radius: 8px !important;
        margin-bottom: var(--spacing-md) !important;
        color: white !important;
        text-align: center !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .main-header h1 {
        margin: 0 !important;
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
    }
    
    .main-header p {
        margin: var(--spacing-sm) 0 0 0 !important;
        font-size: 0.9rem !important;
        opacity: 0.85 !important;
        line-height: 1.2 !important;
    }
    
    /* Session card */
    .session-card {
        background: var(--card-background);
        padding: var(--spacing-md) !important;
        border-radius: 6px !important;
        border-left: 3px solid #667eea;
        margin: var(--spacing-sm) 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: var(--text-color);
        max-width: 100% !important;
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.4rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
        font-size: 0.9rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        padding: 0.3rem 0.8rem !important;
        border-radius: 6px !important;
        font-size: 0.85rem !important;
        line-height: 1.2 !important;
        margin: var(--spacing-sm) 0 !important;
    }
    
    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        padding: var(--spacing-md) !important;
        background-color: var(--card-background) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        margin: var(--spacing-sm) 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        margin: 0 0 var(--spacing-sm) 0 !important;
        padding: 0 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stTextInput {
        margin: var(--spacing-sm) 0 !important;
        padding: 0 !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        margin: var(--spacing-sm) 0 !important;
        padding: 0 !important;
        font-size: 0.75rem !important;
    }
    
    /* Conversation list */
    #convo-list .element-container { margin: 0 !important; padding: 0 !important; }
    #convo-list .stButton { margin: var(--spacing-sm) 0 !important; }
    #convo-list .stButton > button { margin: 0 !important; }
    #convo-list .conv-title { margin: 0 !important; }
    #convo-list .conv-group { display: flex; flex-direction: column; gap: var(--spacing-sm) !important; margin: 0 !important; padding: 0 !important; }
    #convo-list .conv-row { margin: 0 !important; padding: 0 !important; }
    #convo-list .conv-row [data-testid="column"] { padding: 0 !important; margin: 0 !important; }
    #convo-list .conv-title .stButton > button {
        box-shadow: none !important;
        background: var(--card-background) !important;
        color: var(--text-color) !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.8rem !important;
        font-size: 0.9rem !important;
    }
    #convo-list .conv-actions .stButton > button {
        padding: 0.2rem 0.5rem !important;
        font-size: 0.8rem !important;
        border-radius: 4px !important;
    }
    #convo-list .conv-actions { margin: 0 !important; margin-top: var(--spacing-sm) !important; }
    
    /* Info box */
    .info-box {
        background: var(--info-box-bg);
        border-left: 3px solid var(--info-box-border);
        padding: var(--spacing-sm) var(--spacing-md) !important;
        border-radius: 6px;
        margin: var(--spacing-md) 0 !important;
        color: var(--text-color);
        font-size: 0.9rem !important;
        text-align: center !important;
        line-height: 1.4 !important;
    }
    
    /* Status indicator */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: var(--spacing-sm);
    }
    
    .status-active { background: #00b894; animation: pulse 2s infinite; }
    .status-inactive { background: #ddd; }
    
    @keyframes pulse { 0%{opacity:1} 50%{opacity:0.5} 100%{opacity:1} }
    
    /* Dark mode overrides */
    .stApp[data-theme="dark"] .stAlert {
        background-color: var(--card-background) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    .stApp[data-theme="dark"] .stMarkdown { color: var(--text-color) !important; }
    .stApp[data-theme="dark"] .stText { color: var(--text-color) !important; }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="success"] {
        background-color: #22543d !important;
        color: #9ae6b4 !important;
        border: 1px solid #38a169 !important;
    }
    .stApp[data-theme="dark"] .stAlert[data-baseweb="notification"][data-severity="error"] {
        background-color: #742a2a !important;
        color: #feb2b2 !important;
        border: 1px solid #e53e3e !important;
    }
    
    /* Chat bubbles */
    [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] { display: none !important; }
    .stChatMessage[data-testid="user-message"] {
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-end !important;
        margin: var(--spacing-sm) 0 !important;
    }
    .stChatMessage[data-testid="assistant-message"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        margin: var(--spacing-sm) 0 !important;
    }
    .stChatMessage[data-testid="user-message"] .stMarkdown {
        background: #667eea !important;
        color: #fff !important;
        padding: var(--spacing-sm) var(--spacing-md) !important;
        border-radius: 10px 10px 3px 10px !important;
        max-width: 70% !important;
        margin-left: auto !important;
    }
    .stChatMessage[data-testid="assistant-message"] .stMarkdown {
        background: var(--card-background) !important;
        color: var(--text-color) !important;
        padding: var(--spacing-sm) var(--spacing-md) !important;
        border-radius: 10px 10px 10px 3px !important;
        max-width: 70% !important;
        margin-right: auto !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Chat input */
    .stChatInput {
        background: var(--background-color) !important;
        padding: var(--spacing-sm) !important;
    }
    .stChatInput > div {
        background: var(--background-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
        padding: var(--spacing-sm) !important;
    }
    .stChatInput textarea, .stChatInput input {
        font-size: 0.95rem !important;
        color: var(--text-color) !important;
        padding: var(--spacing-sm) !important;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Helpers
def _load():
    try: return json.load(open("conversations.json","r",encoding="utf-8")) if os.path.exists("conversations.json") else {}
    except: return {}

def _save(d):
    try: json.dump(d, open("conversations.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    except: pass

def calculate_cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

S = st.session_state
if "conversations" not in S: S.conversations = _load()
if "cur" not in S: S.cur = next(iter(S.conversations), None)
if "hf" not in S: S.hf = os.getenv("HF_TOKEN", "") or _fallback_read_hf_token()
if S.hf:
    os.environ["HF_TOKEN"] = S.hf
if "rename_id" not in S: S.rename_id = None
if "rename_value" not in S: S.rename_value = ""
if "confirm_delete_id" not in S: S.confirm_delete_id = None
if "embedding_model" not in S: 
    try:
        S.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except:
        st.warning("SentenceTransformer model not loaded. RAG functionality may not work.")
VERSION = "ui-rename-delete+token-ctrl v4"

with st.sidebar:
    st.markdown('<div><h3>🤖 GPT-OSS-20B Chat</h3></div>', unsafe_allow_html=True)
    
    # PDF Upload Section - Moved to top of sidebar
    st.markdown("### 📄 PDF Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], help="Upload a PDF to enable document-based Q&A")
    if uploaded_file is not None:
        if "pdf_name" not in S or S.pdf_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                try:
                    pdf_reader = PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    # Simple chunking with overlap
                    chunk_size = 1000
                    overlap = 200
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunks.append(text[start:end])
                        start = end - overlap
                    
                    embeddings = S.embedding_model.encode(chunks, show_progress_bar=False)
                    S.vector_store = embeddings  # Store embeddings directly
                    S.chunks = chunks
                    S.pdf_name = uploaded_file.name
                    st.success(f"✅ PDF '{uploaded_file.name}' processed successfully! You can now ask questions about it.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        else:
            st.info(f"📖 Using: {S.pdf_name}")
    
    # RAG Toggle
    if "vector_store" in S and S.vector_store is not None:
        use_rag = st.checkbox("Enable PDF Q&A", value=True, help="Use the uploaded PDF as context for answers")
    else:
        use_rag = False
    
    level = st.selectbox("Reasoning Level", ["Low","Medium","High"], index=1, help="Select the reasoning complexity for responses.")
    
    if not S.hf:
        st.markdown("### 🔑 API Token")
        token_input = st.text_input("HF Token", value=S.hf, type="password", help="Paste your Hugging Face Inference token.")
        if token_input != S.hf:
            S.hf = token_input.strip()
            if S.hf:
                os.environ["HF_TOKEN"] = S.hf
        st.caption(f"Token: {'Set' if S.hf else 'Not set'}")
        save_env = st.checkbox("Save token to .env (local only)")
        if save_env and S.hf and st.button("Save HF_TOKEN", use_container_width=True):
            try:
                env_path = ".env"
                lines = []
                if os.path.exists(env_path):
                    with open(env_path, "r", encoding="utf-8") as f:
                        lines = f.read().splitlines()
                lines = [ln for ln in lines if not ln.strip().startswith("HF_TOKEN=")]
                lines.append(f"HF_TOKEN={S.hf}")
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                st.success("Saved HF_TOKEN to .env")
            except Exception as e:
                st.error(f"Failed to save .env: {e}")
        if st.button("Reload .env", use_container_width=True):
            load_dotenv(override=True)
            S.hf = os.getenv("HF_TOKEN", "") or S.hf
            if S.hf:
                os.environ["HF_TOKEN"] = S.hf
            st.rerun()
    
    st.markdown("### 💬 Chat")
    if st.button("➕ New Chat", use_container_width=True):
        i = str(uuid.uuid4()); S.conversations[i] = {"title":"New Chat","messages":[]}; S.cur = i; _save(S.conversations); st.rerun()
    st.markdown('<div><h4>Conversations</h4></div>', unsafe_allow_html=True)
    st.markdown('<div id="convo-list">', unsafe_allow_html=True)
    if not S.conversations: st.markdown('<div class="info-box">No conversations yet. Start a new chat!</div>', unsafe_allow_html=True)
    for i, c in list(S.conversations.items()):
        st.markdown('<div class="conv-group">', unsafe_allow_html=True)
        st.markdown('<div class="conv-title">', unsafe_allow_html=True)
        if st.button(c.get("title","New Chat"), key=f"sel_{i}", use_container_width=True): S.cur = i; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="conv-row conv-actions">', unsafe_allow_html=True)
        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Rename", key=f"ren_{i}", use_container_width=True):
                S.rename_id = i; S.rename_value = c.get("title","New Chat")
        with col_right:
            if st.button("Delete", key=f"del_{i}", use_container_width=True):
                S.conversations.pop(i, None)
                S.cur = next(iter(S.conversations), None)
                _save(S.conversations)
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        if S.rename_id == i:
            new_title = st.text_input("Rename conversation", value=S.rename_value, key=f"ren_input_{i}")
            rcol1, rcol2 = st.columns(2)
            if rcol1.button("Save", key=f"ren_save_{i}", use_container_width=True):
                title = (new_title or "").strip() or c.get("title","New Chat")
                S.conversations[i]["title"] = title
                _save(S.conversations)
                S.rename_id = None; S.rename_value = ""
                st.rerun()
            if rcol2.button("Cancel", key=f"ren_cancel_{i}", use_container_width=True):
                S.rename_id = None; S.rename_value = ""
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🤖 GPT-OSS-20B Chat</h1>
    <p>Conversational AI powered by open-source 20B model</p>
</div>
""", unsafe_allow_html=True)

if not S.hf:
    st.markdown('<div class="info-box">⚠️ Set HF_TOKEN in the sidebar to enable chatting.</div>', unsafe_allow_html=True)
    client = None
else:
    try:
        client = InferenceClient("openai/gpt-oss-20b", token=S.hf)
    except Exception as e:
        client = None
        st.error(str(e))

# Show PDF status
if "pdf_name" in S and S.pdf_name:
    st.info(f"📄 PDF loaded: {S.pdf_name} | {'✅ RAG Enabled' if use_rag else '❌ RAG Disabled'}")

msgs = S.conversations.get(S.cur, {}).get("messages", []) if S.cur else []
for m in msgs:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Type your message here..."):
    if not S.cur:
        S.cur = str(uuid.uuid4()); S.conversations[S.cur] = {"title":"New Chat","messages":[]}
    msgs.append({"role":"user","content":prompt}); S.conversations[S.cur]["messages"] = msgs
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            if client is None:
                st.error("HF_TOKEN missing or invalid. Add it in the sidebar and try again.")
            else:
                sys = {"Low":"Reasoning: low","Medium":"Reasoning: medium","High":"Reasoning: high"}[level]
                temp_msgs = [{"role":"system","content":f"You are a helpful assistant. {sys}"}]
                temp_msgs.extend(msgs)
                
                if use_rag and "vector_store" in S and S.vector_store is not None:
                    try:
                        query_embedding = S.embedding_model.encode([prompt])
                        # Brute-force cosine similarity search
                        similarities = [calculate_cosine_similarity(query_embedding[0], emb) for emb in S.vector_store]
                        top_k = min(3, len(S.vector_store))  # Ensure we don't exceed available chunks
                        relevant_indices = np.argsort(similarities)[-top_k:][::-1]  # Get top 3 indices
                        if relevant_indices.size > 0:
                            context = "\n\n".join(S.chunks[i] for i in relevant_indices)
                            temp_msgs[-1]["content"] = f"Use the following context from the document to answer the question:\n\n{context}\n\nQuestion: {prompt}\n\nAnswer based only on the provided context."
                        else:
                            temp_msgs[-1]["content"] = prompt
                    except Exception as e:
                        st.warning(f"RAG processing error: {e}. Using regular chat.")
                        temp_msgs[-1]["content"] = prompt
                else:
                    temp_msgs[-1]["content"] = prompt
                
                resp = client.chat_completion(messages=temp_msgs, temperature=0.7, max_tokens=1000, stream=True)
                out, box = "", st.empty()
                for ch in resp:
                    t = getattr(getattr(ch.choices[0],"delta",object()),"content",None)
                    if t is None and hasattr(ch,"generated_text"): out = ch.generated_text
                    elif t: out += t
                    box.markdown(out + "▌")
                box.markdown(out)
                
                # Update the actual message (without context)
                msgs[-1]["content"] = prompt
                msgs.append({"role":"assistant","content": out})
                
                if len(msgs)==2: 
                    S.conversations[S.cur]["title"] = msgs[0]["content"][:30]+("..." if len(msgs[0]["content"])>30 else "")
                S.conversations[S.cur]["messages"] = msgs
                _save(S.conversations)
        except Exception as e: 
            st.error(str(e))

with st.container():
    st.markdown('<div id="clear-chat">', unsafe_allow_html=True)
    if S.cur and st.button("🗑️ Clear Current Chat", use_container_width=True):
        S.conversations[S.cur]["messages"] = []; _save(S.conversations); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)