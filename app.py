"""
app.py  —  Compliance Chatbot
─────────────────────────────
Flow:
  1. User asks a question
  2. RAG retrieves relevant legal chunks from Supabase
  3. Fine-tuned TinyLlama answers using those chunks
  4. If TinyLlama not found → Gemini 1.5 Flash answers instead

Run ingestion first (once):       python ingest.py
Run fine-tuning first (once):     python finetune.py
Then launch:                       streamlit run app.py
"""

import os
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from supabase import create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
# These values come from your .env file
PROJECT_ID          = os.getenv("PROJECT_ID")
SUPABASE_URL        = f"https://{PROJECT_ID}.supabase.co"
SUPABASE_KEY        = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
FINETUNED_MODEL_DIR = "./finetuned_model"   # folder produced by finetune.py
EMBED_MODEL         = "all-MiniLM-L6-v2"   # turns text into vectors for search
TOP_K               = 5                     # how many chunks to retrieve per question
MAX_HISTORY_TURNS   = 5                     # how many past messages to remember


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — INIT
# These functions run once and are cached by Streamlit so they don't
# reload every time the user sends a message.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def init_supabase_and_embeddings():
    """
    Connects to Supabase (your vector database)
    and loads the embedding model that converts text → numbers.
    """
    supabase   = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return supabase, embeddings


@st.cache_resource
def init_gemini_client():
    """
    Connects to the Gemini API.
    Only used if the fine-tuned model folder is missing.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — RAG STEP 1: RETRIEVE
# Searches Supabase for the legal text chunks most relevant
# to the user's question.
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(question: str, supabase, embeddings) -> list[Document]:
    """
    R in RAG.
    1. Converts the question into a vector (list of numbers)
    2. Searches Supabase for the 5 most similar vectors
    3. Returns those chunks as Document objects with metadata
    """
    # Step 1: turn the question into a vector
    vector = embeddings.embed_query(question)

    # Step 2: search Supabase using the match_documents SQL function
    try:
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": vector,
                "match_count": TOP_K,
                "match_threshold": 0.5,   # minimum similarity score (0-1)
            },
        ).execute()
    except Exception as e:
        st.sidebar.error(f"Supabase search error: {e}")
        return []

    # Step 3: convert results into Document objects
    docs = []
    for row in response.data or []:
        docs.append(
            Document(
                page_content=row.get("content", row.get("page_content", "")),
                metadata={
                    "file_name":  row.get("file_name", ""),
                    "source_url": row.get("source_url", ""),
                    **(row.get("metadata") or {}),
                },
            )
        )
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FINE-TUNED LOCAL MODEL (TinyLlama)
# Loads the model produced by finetune.py.
# ─────────────────────────────────────────────────────────────────────────────

def _finetuned_model_exists() -> bool:
    """Checks if finetune.py has been run and produced output files."""
    p = Path(FINETUNED_MODEL_DIR)
    return p.exists() and (
        any(p.glob("*.safetensors")) or any(p.glob("adapter_model*"))
    )


@st.cache_resource
def init_local_model():
    """
    Loads the fine-tuned TinyLlama model with its LoRA adapter.
    This matches exactly how finetune.py saved it.
    Heavy operation — cached so it only runs once.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel, PeftConfig

    # Read the adapter config to find which base model was used
    config     = PeftConfig.from_pretrained(FINETUNED_MODEL_DIR)
    base_model = config.base_model_name_or_path  # TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Load tokenizer saved by finetune.py
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the base TinyLlama model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        device_map="auto",        # uses GPU if available, otherwise CPU
        trust_remote_code=True,
    )

    # Apply the LoRA adapter trained by finetune.py on top of the base model
    model = PeftModel.from_pretrained(model, FINETUNED_MODEL_DIR)
    model.eval()   # inference mode, not training mode

    # Wrap in a pipeline for easy text generation
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.2,           # low = more focused/deterministic answers
        repetition_penalty=1.1,    # discourages repeating the same words
        do_sample=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PROMPT BUILDERS
# RAG Step 2 (Augment): combines the retrieved chunks + question into
# a prompt the model can understand.
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_local(context: str, question: str, history: list) -> str:
    """
    Builds the prompt for TinyLlama.
    Uses the exact same token format as finetune.py so the model
    recognises the structure it was trained on.
    """
    system_msg = (
        "You are a compliance assistant specialised in Tunisian labour law and "
        "regulatory texts. Answer strictly based on the legal documents provided."
    )

    # Add previous conversation turns so the model has context
    history_str = ""
    for turn in history:
        history_str += (
            f"<|user|>\n{turn['user']}</s>\n"
            f"<|assistant|>\n{turn['assistant']}</s>\n"
        )

    # Final prompt: system + history + retrieved chunks + question
    return (
        f"<|system|>\n{system_msg}</s>\n"
        f"{history_str}"
        f"<|user|>\n"
        f"Relevant legal excerpts:\n\n{context}\n\n"
        f"Question: {question}</s>\n"
        f"<|assistant|>\n"
    )


def build_prompt_gemini(context: str, question: str, history: list) -> str:
    """
    Builds the prompt for Gemini.
    Plain text — Gemini doesn't need special tokens.
    """
    system_msg = (
        "You are a compliance assistant specialised in Tunisian labour law and "
        "regulatory texts. Answer questions strictly based on the provided legal "
        "document excerpts. If the answer is not in the context, say so clearly. "
        "Always cite the source document when possible. "
        "Respond in the same language as the user's question (French, Arabic, or English)."
    )

    history_str = ""
    for turn in history:
        history_str += f"\nUser: {turn['user']}\nAssistant: {turn['assistant']}"

    return (
        f"{system_msg}\n\n"
        f"{history_str}\n\n"
        f"Relevant legal document excerpts:\n{context}\n\n"
        f"Question: {question}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GENERATION
# RAG Step 3 (Generate): sends the prompt to the model and gets an answer.
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer_local(pipe, prompt: str) -> str:
    """
    Runs TinyLlama and extracts only the new answer text.
    The pipeline returns the full prompt + answer, so we
    strip the prompt prefix and stop at TinyLlama's stop tokens.
    """
    output = pipe(prompt)
    # Remove the prompt itself — we only want what the model added
    answer = output[0]["generated_text"][len(prompt):].strip()

    # Cut off at TinyLlama stop tokens if they appear
    for stop in ["</s>", "<|eot_id|>", "<|user|>", "<|system|>"]:
        if stop in answer:
            answer = answer.split(stop)[0].strip()
    return answer


def generate_answer_gemini(context: str, question: str, history: list) -> str:
    """Sends the prompt to Gemini and returns its response."""
    client   = init_gemini_client()
    prompt   = build_prompt_gemini(context, question, history)
    response = client.generate_content(prompt)
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FULL RAG PIPELINE
# Ties everything together: Retrieve → Augment → Generate
# ─────────────────────────────────────────────────────────────────────────────

def ask(question: str, history: list, supabase, embeddings) -> tuple[str, list]:
    """
    The main function called when a user sends a message.

    1. RETRIEVE  — search Supabase for relevant legal chunks
    2. AUGMENT   — build a prompt combining chunks + question + history
    3. GENERATE  — TinyLlama answers (or Gemini if model not found)
    """

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    docs = retrieve(question, supabase, embeddings)

    if not docs:
        return (
            "⚠️ No relevant documents found. "
            "Make sure `ingest.py` was run and your Supabase `doc_chunks` table has data.",
            [],
        )

    # ── Step 2: Augment — build context and collect source file names ─────────
    context_parts = []
    sources       = []
    for doc in docs:
        context_parts.append(doc.page_content)
        fname = doc.metadata.get("file_name", "")
        url   = doc.metadata.get("source_url", "")
        if fname and fname not in [s[0] for s in sources]:
            sources.append((fname, url))

    # Join all retrieved chunks into one context block
    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: Generate ──────────────────────────────────────────────────────
    try:
        if _finetuned_model_exists():
            # Primary: use the fine-tuned TinyLlama (trained on your data)
            pipe   = init_local_model()
            prompt = build_prompt_local(context, question, history)
            answer = generate_answer_local(pipe, prompt)
        else:
            # Fallback: use Gemini if fine-tuned model folder is missing
            answer = generate_answer_gemini(context, question, history)

    except Exception as e:
        answer = f"❌ Error generating response: {e}"

    return answer, sources


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — STREAMLIT UI
# The chat interface the user sees and interacts with.
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Compliance Chatbot",
    page_icon="⚖️",
    layout="centered",
)

# ── Sidebar — shows which model is active ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    if _finetuned_model_exists():
        st.success("🧠 Fine-tuned TinyLlama active\n`./finetuned_model/`")
    else:
        st.warning(
            "⚡ Fine-tuned model not found\n"
            "Falling back to **Gemini 1.5 Flash**\n\n"
            "Run `python finetune.py` to enable the local model."
        )
    st.caption(f"Embeddings: `{EMBED_MODEL}`")
    st.caption(f"RAG top-k: `{TOP_K}`")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.title("⚖️ Compliance Assistant")
st.caption("Tunisian labour law & regulatory texts · RAG + Fine-tuned TinyLlama / Gemini fallback")

# Session state stores the full conversation so it survives page reruns
if "messages" not in st.session_state:
    st.session_state.messages = []   # full chat history for display
if "history" not in st.session_state:
    st.session_state.history = []    # last 5 turns sent to the model

# Render all previous messages on the screen
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources"):
                for fname, url in msg["sources"]:
                    st.markdown(f"- [{fname}]({url})" if url else f"- {fname}")

# ── Handle new user message ───────────────────────────────────────────────────
if question := st.chat_input("Ask a compliance question…", key="main_input"):

    # Init connections (cached — only actually runs on first message)
    supabase, embeddings = init_supabase_and_embeddings()

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and show the assistant's answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            answer, sources = ask(question, st.session_state.history, supabase, embeddings)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for fname, url in sources:
                    st.markdown(f"- [{fname}]({url})" if url else f"- {fname}")

    # Save to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.history.append({"user": question, "assistant": answer})

    # Keep only the last 5 turns in the model's context window
    if len(st.session_state.history) > MAX_HISTORY_TURNS:
        st.session_state.history = st.session_state.history[-MAX_HISTORY_TURNS:]