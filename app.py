import os
import streamlit as st
from supabase import create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
 
load_dotenv()
 
# --- CONFIG ---
PROJECT_ID = os.getenv("PROJECT_ID")
SUPABASE_URL = f"https://{PROJECT_ID}.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
LLM_MODEL = "meta-llama/Llama-3-8B-Instruct"
 
# --- INIT (cached so it only runs once) ---
@st.cache_resource
def init_retriever():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="doc_chunks",
        query_name="match_documents",
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})
 
@st.cache_resource
def init_llm_client():
    return InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
 
retriever = init_retriever()
llm = init_llm_client()
 
# --- RAG LOGIC ---
def build_prompt(context: str, question: str, history: list) -> str:
    system_msg = (
        "You are a compliance assistant specialized in Tunisian labor law and regulatory texts. "
        "Answer questions strictly based on the provided legal document excerpts. "
        "If the answer is not found in the context, say so clearly. "
        "Always cite the source document name when possible. "
        "Respond in the same language as the user's question (French or Arabic or English)."
    )
 
    # Build conversation history string
    history_str = ""
    for turn in history:
        history_str += f"\nUser: {turn['user']}\nAssistant: {turn['assistant']}"
 
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_msg}<|eot_id|>
{history_str}
<|start_header_id|>user<|end_header_id|>
Here are relevant excerpts from legal documents:
 
{context}
 
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return prompt
 
def ask(question: str, history: list) -> tuple[str, list[str]]:
    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)
 
    # 2. Build context + collect sources
    context_parts = []
    sources = []
    for doc in docs:
        context_parts.append(doc.page_content)
        fname = doc.metadata.get("file_name", "")
        url = doc.metadata.get("source_url", "")
        if fname and fname not in sources:
            sources.append((fname, url))
 
    context = "\n\n---\n\n".join(context_parts)
 
    # 3. Call LLM
    prompt = build_prompt(context, question, history)
    response = llm.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.2n,
        repetition_penalty=1.1,
        stop_sequences=["<|eot_id|>", "<|start_header_id|>"],
    )
 
    return response.strip(), sources
 
# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Compliance Chatbot",
    page_icon="⚖️",
    layout="centered",
)
 
st.title("⚖️ Compliance Assistant")
st.caption("Ask questions about Tunisian labor law and regulatory texts.")
 
# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": ..., "sources": [...]}
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"user": ..., "assistant": ...} for prompt context
 
# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for fname, url in msg["sources"]:
                    if url:
                        st.markdown(f"- [{fname}]({url})")
                    else:
                        st.markdown(f"- {fname}")
 
# Chat input
if question := st.chat_input("Ask a compliance question..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
 
    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            answer, sources = ask(question, st.session_state.history)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for fname, url in sources:
                    if url:
                        st.markdown(f"- [{fname}]({url})")
                    else:
                        st.markdown(f"- {fname}")
 
    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.history.append({"user": question, "assistant": answer})
 
    # Keep history to last 5 turns to avoid prompt bloat
    if len(st.session_state.history) > 5:
        st.session_state.history = st.session_state.history[-5:]
