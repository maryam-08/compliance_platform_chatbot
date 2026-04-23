import os
import fitz  # This is PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
from dotenv import load_dotenv
from httpx import Timeout
from supabase.lib.client_options import ClientOptions
load_dotenv()
engine = RapidOCR()

def ocr_pdf(file_path):
    """Custom function to turn a scanned PDF into text using PyMuPDF + RapidOCR"""
    doc = fitz.open(file_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Turn the page into an image (no Poppler needed!)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        img_bytes = pix.tobytes()
        
        # Run OCR
        result, _ = engine(img_bytes)
        if result:
            page_text = "\n".join([line[1] for line in result])
            full_text.append(page_text)
    
    return "\n\n".join(full_text)

# --- CONFIG ---
PROJECT_ID = os.getenv("PROJECT_ID")
SUPABASE_URL = f"https://{PROJECT_ID}.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
# Create a custom timeout (wait 60 seconds instead of the default 5)
custom_timeout = Timeout(60.0, read=60.0, connect=60.0)
# We set the timeout for Postgrest (the database part)
from supabase import create_client

# Back to basics - No complex options needed
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

DATA_PATH = "./data/"
BUCKET_NAME = "legal_docs"
BASE_STORAGE_URL = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/"

# --- PROCESS ---
for filename in os.listdir(DATA_PATH):
    if filename.endswith(".pdf"):
        file_path = os.path.join(DATA_PATH, filename)
        print(f"🧐 Starting OCR on: {filename}...")
        
        # 1. Get Text via OCR
        raw_text = ocr_pdf(file_path)
        
        if not raw_text.strip():
            print(f"⚠️ Warning: No text found in {filename}")
            continue

        # 2. Create LangChain Document
        public_url = BASE_STORAGE_URL + filename
        doc = Document(
            page_content=raw_text,
            metadata={"source_url": public_url, "file_name": filename}
        )

        # 3. Chunk and Upload
        chunks = text_splitter.split_documents([doc])
        print(f"📤 Pushing {len(chunks)} chunks to Supabase in baches...")
        # Split chunks into batches of 5
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            SupabaseVectorStore.from_documents(
                batch,
                embeddings_model,
                client=supabase,
                table_name="doc_chunks",
                query_name="match_documents"
            )
            print(f"   ... Sent batch {i//batch_size + 1}")

        print(f"✅ Fully Indexed: {filename}")
        