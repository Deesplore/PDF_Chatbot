
# from docx import Document as DocxDocument
# import streamlit as st
# import PyPDF2
# import os
# from PyPDF2 import PdfReader
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema.document import Document
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import google.generativeai as genai
# import pdfplumber
# load_dotenv()

# api_key = os.getenv("GOOGLE_API_KEY")
# if api_key is None:
#     raise ValueError("GOOGLE_API_KEY environment variable is not set")

# genai.configure(api_key=api_key)


# def extract_text_from_files(uploaded_files):
#     text = ""

#     for file in uploaded_files:
#         file_type = file.name.split(".")[-1].lower()

#         if file_type == "pdf":
#             text += extract_pdf_text(file)

#         elif file_type == "docx":
#             text += extract_docx_text(file)

#         elif file_type == "txt":
#             text += extract_txt_text(file)

#         else:
#             st.warning(f"Unsupported file type: {file.name}")

#     return text     

# def extract_docx_text(file):
#     text = ""

#     doc = DocxDocument(file)

#     # Paragraphs
#     for para in doc.paragraphs:
#         if para.text.strip():
#             text += para.text.strip() + "\n"

#     # Tables
#     for table in doc.tables:
#         text += "\n"
#         for row in table.rows:
#             row_text = [cell.text.strip() for cell in row.cells]
#             text += " | ".join(row_text) + "\n"

#     return text

# def extract_txt_text(file):
#     return file.read().decode("utf-8").strip()

# def extract_pdf_text(file):
#     text = ""
#     with pdfplumber.open(file) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text(layout=True)
#             if page_text:
#                 text += page_text + "\n"
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     return vector_store

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "Not found in document", Please provide confidence score alss for the answer given just ,don't provide the wrong answer or generalize the answer only answer from the given provided context\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="models/gemma-3-27b-it", temperature=0.6)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_question, vector_store):
#     retriever = vector_store.as_retriever()
#     docs = retriever.get_relevant_documents(user_question)

#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": user_question}, return_only_outputs=True
#     )

#     # Record chat history in session state
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     # Insert the latest chat at the beginning
#     st.session_state.chat_history.insert(0, {
#         "user": user_question,
#         "bot": response["output_text"]
#     })

#     st.write("Reply: ", response["output_text"])

# EXTRACTION_QUESTION = """
# Extract everything in JSON. If a field is not found, return null.
# {
#   "shipment_id": "",
#   "shipper": "",
#   "consignee": "",
#   "pickup_datetime": "",
#   "delivery_datetime": "",
#   "equipment_type": "",
#   "mode": "",
#   "rate": "",
#   "currency": "",
#   "weight": "",
#   "carrier_name": ""
# }
# Please no not give explanation strictly guve answer in json Do not generalize any answer if any answer is not given just give null for that value.
# """
# def extract_json_from_pdf():
#     if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
#         st.warning("Please upload and process PDF files first!")
#         return

#     vector_store = st.session_state.vector_store
#     retriever = vector_store.as_retriever()
#     docs = retriever.get_relevant_documents(EXTRACTION_QUESTION)

#     chain = get_conversational_chain()
#     response = chain(
#         {"input_documents": docs, "question": EXTRACTION_QUESTION},
#         return_only_outputs=True
#     )

#     st.subheader("Extracted JSON")
#     st.code(response["output_text"], language="json")

# def main():
#     st.set_page_config(page_title="Document Reader")
#     st.header("Document Reader")

#     if 'vector_store' not in st.session_state:
#         st.session_state.vector_store = None

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question and st.session_state.vector_store:
#         user_input(user_question, st.session_state.vector_store)

#     with st.sidebar:
#         st.title("Menu")
#         uploaded_files = st.file_uploader( "Upload PDF / DOCX / TXT Files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             if uploaded_files:
#                 with st.spinner("Processing..."):
#                     raw_text = extract_text_from_files(uploaded_files)

#                     text_chunks = get_text_chunks(raw_text)
#                     st.session_state.vector_store = get_vector_store(text_chunks)
#                     st.session_state.raw_text = raw_text

#                     st.success("Done")
#             else:
#                 st.warning("Please upload at least one file")

#         # Display the uploaded PDF content
#         # st.subheader("Uploaded PDF Content")
#         # if 'raw_text' in st.session_state and st.session_state.raw_text:
#             # st.text_area("PDF Content", value=st.session_state.raw_text, height=400)

                
#         if st.sidebar.button("Extract JSON from PDF"):
#             with st.spinner("Extracting JSON..."):
#                 extract_json_from_pdf()


# if __name__ == "__main__":
#     main()



from docx import Document as DocxDocument
import streamlit as st
import os
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
import pdfplumber
# load_dotenv()
# Make sure the environment variable is loaded correctly
GOOGLE_API_KEY= st.secrets["GOOGLE_API_KEY"]
load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.45   # cosine distance; lower = more similar in FAISS L2
LOW_CONFIDENCE_CUTOFF = 30    # if parsed confidence < this, refuse to show answer

api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
genai.configure(api_key=api_key)

# ─────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TMS Doc Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0f1117; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 100%);
    }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e2e8f0; }

    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: -0.5px;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Answer card */
    .answer-card {
        background: #1e2235;
        border: 1px solid #2d3748;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }

    .answer-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }

    .answer-text {
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* Confidence badge */
    .badge-high {
        display: inline-block;
        background: #064e3b;
        color: #34d399;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        border: 1px solid #34d399;
    }
    .badge-mid {
        display: inline-block;
        background: #451a03;
        color: #fb923c;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        border: 1px solid #fb923c;
    }
    .badge-low {
        display: inline-block;
        background: #450a0a;
        color: #f87171;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        border: 1px solid #f87171;
    }

    /* Source chunk */
    .source-chunk {
        background: #141824;
        border: 1px solid #2d3748;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .source-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    /* Chat history item */
    .history-item {
        border-bottom: 1px solid #2d3748;
        padding: 0.8rem 0;
        margin: 0.3rem 0;
    }
    .history-q { color: #38bdf8; font-size: 0.85rem; font-weight: 600; }
    .history-a { color: #94a3b8; font-size: 0.82rem; margin-top: 4px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #141824;
        border-right: 1px solid #2d3748;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* Buttons */
    .stButton > button {
        background: #0369a1;
        color: white;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.5rem 1.2rem;
        transition: background 0.2s;
        width: 100%;
    }
    .stButton > button:hover { background: #0284c7; }

    /* Text input */
    .stTextInput > div > div > input {
        background: #1e2235;
        color: #e2e8f0;
        border: 1px solid #2d3748;
        border-radius: 6px;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; }

    /* Refusal card */
    .refusal-card {
        background: #1a0f0f;
        border: 1px solid #7f1d1d;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        color: #fca5a5;
        font-size: 0.9rem;
    }

    div[data-testid="stCodeBlock"] {
        background: #141824 !important;
        border: 1px solid #2d3748;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────
def extract_text_from_files(uploaded_files):
    text = ""
    for file in uploaded_files:
        file_type = file.name.split(".")[-1].lower()
        if file_type == "pdf":
            text += extract_pdf_text(file)
        elif file_type == "docx":
            text += extract_docx_text(file)
        elif file_type == "txt":
            text += extract_txt_text(file)
        else:
            st.warning(f"Unsupported file type: {file.name}")
    return text

def extract_docx_text(file):
    text = ""
    doc = DocxDocument(file)
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text.strip() + "\n"
    for table in doc.tables:
        text += "\n"
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells]
            text += " | ".join(row_text) + "\n"
    return text

def extract_txt_text(file):
    return file.read().decode("utf-8").strip()

def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(layout=True)
            if page_text:
                text += page_text + "\n"
    return text


# ─────────────────────────────────────────────
# CHUNKING & VECTOR STORE
# ─────────────────────────────────────────────
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# ─────────────────────────────────────────────
# CONFIDENCE PARSING
# ─────────────────────────────────────────────
def parse_confidence(text: str) -> int:
    """
    Extract the confidence integer the LLM returns.
    Looks for patterns like: Confidence: 87 | confidence score: 72% | [confidence: 90]
    Falls back to None if not found.
    """
    patterns = [
        r"confidence[^\d]{0,15}(\d{1,3})",
        r"(\d{1,3})\s*(?:%|percent)\s*confident",
        r"\[(\d{1,3})\]",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                return val
    return None

def split_answer_and_confidence(raw: str):
    """Return (clean_answer, confidence_int_or_None)."""
    confidence = parse_confidence(raw)
    # Strip the confidence line from the displayed answer
    clean = re.sub(
        r"(confidence\s*(?:score)?[:\s\-]*\d{1,3}\s*%?\.?)",
        "",
        raw,
        flags=re.IGNORECASE
    ).strip()
    return clean, confidence

def confidence_badge_html(score: int) -> str:
    if score is None:
        return '<span class="badge-mid">Confidence: N/A</span>'
    elif score >= 70:
        return f'<span class="badge-high">Confidence: {score}%</span>'
    elif score >= 40:
        return f'<span class="badge-mid">Confidence: {score}%</span>'
    else:
        return f'<span class="badge-low">Confidence: {score}%</span>'


# ─────────────────────────────────────────────
# RETRIEVAL WITH SIMILARITY GUARD
# ─────────────────────────────────────────────
def retrieve_with_scores(question: str, vector_store):
    """
    Returns (docs, scores, passed_threshold).
    FAISS returns L2 distances — lower is better.
    We convert to a 0-100 similarity score for display.
    """
    results = vector_store.similarity_search_with_score(question, k=4)
    docs = [r[0] for r in results]
    raw_scores = [r[1] for r in results]

    # Best (lowest) L2 distance
    best_score = min(raw_scores)
    passed = best_score <= SIMILARITY_THRESHOLD

    # Convert L2 distance → 0-100 similarity for display (heuristic)
    display_scores = [max(0, round((1 - s / 2) * 100)) for s in raw_scores]

    return docs, display_scores, passed, best_score


# ─────────────────────────────────────────────
# LLM CHAIN
# ─────────────────────────────────────────────
def get_conversational_chain():
    prompt_template = """
You are a logistics document assistant. Answer ONLY from the provided context.

Rules:
- If the answer is clearly present: answer precisely and return a confidence score (0-100).
- If the answer is partially present: answer what you can, note what's missing, lower confidence.
- If the answer is NOT in the context at all: respond exactly with "Not found in document."
- Do NOT generalize or use outside knowledge.
- Always end your response with exactly this line:
  Confidence: <number between 0 and 100>

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="models/gemma-3-27b-it", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# ─────────────────────────────────────────────
# Q&A HANDLER
# ─────────────────────────────────────────────
def handle_question(user_question: str, vector_store):
    docs, display_scores, passed_threshold, best_l2 = retrieve_with_scores(
        user_question, vector_store
    )

    # ── GUARDRAIL 1: Retrieval similarity too low ──
    if not passed_threshold:
        retrieval_score = display_scores[0] if display_scores else 0
        st.markdown(f"""
        <div class="refusal-card">
            ⚠️ <strong>Low retrieval confidence — answer refused.</strong><br><br>
            The best matching chunk in your document has a similarity score of
            <strong>{retrieval_score}%</strong>, which is below the minimum threshold.
            This question likely refers to content not present in the uploaded document.
        </div>
        """, unsafe_allow_html=True)

        _record_history(user_question, "⚠️ Refused — low retrieval similarity.", None)
        return

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    raw_answer = response["output_text"]

    # ── GUARDRAIL 2: "Not found in document" passthrough ──
    if "not found in document" in raw_answer.lower():
        st.markdown("""
        <div class="refusal-card">
            🔍 <strong>Not found in document.</strong><br>
            The uploaded document does not contain information to answer this question.
        </div>
        """, unsafe_allow_html=True)
        _record_history(user_question, "Not found in document.", 0)
        return

    clean_answer, confidence = split_answer_and_confidence(raw_answer)

    # ── GUARDRAIL 3: Parsed confidence too low ──
    if confidence is not None and confidence < LOW_CONFIDENCE_CUTOFF:
        st.markdown(f"""
        <div class="refusal-card">
            ⚠️ <strong>Answer confidence too low ({confidence}%) — answer withheld.</strong><br><br>
            The model was not confident enough in its answer based on the document content.
            Try rephrasing your question or check if this information exists in the document.
        </div>
        """, unsafe_allow_html=True)
        _record_history(user_question, f"Withheld — confidence {confidence}%", confidence)
        return

    # ── DISPLAY ANSWER ──
    badge = confidence_badge_html(confidence)
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-label">Answer &nbsp;·&nbsp; {badge}</div>
        <div class="answer-text">{clean_answer}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── DISPLAY SOURCE CHUNKS ──
    with st.expander("📄 Source Chunks Used", expanded=False):
        st.markdown('<div class="source-header">Retrieved context passages</div>', unsafe_allow_html=True)
        for i, (doc, score) in enumerate(zip(docs, display_scores)):
            st.markdown(f"""
            <div class="source-chunk">
                <span style="color:#38bdf8; font-size:0.7rem;">CHUNK {i+1} · Similarity {score}%</span><br><br>
                {doc.page_content}
            </div>
            """, unsafe_allow_html=True)

    _record_history(user_question, clean_answer, confidence)


def _record_history(question, answer, confidence):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.insert(0, {
        "user": question,
        "bot": answer,
        "confidence": confidence
    })


# ─────────────────────────────────────────────
# STRUCTURED EXTRACTION
# ─────────────────────────────────────────────
EXTRACTION_QUESTION = """
Extract the following fields strictly from the document. Return ONLY valid JSON, no explanation, no markdown code fences.
If a field is not found return null for that field.

{
  "shipment_id": null,
  "shipper": null,
  "consignee": null,
  "pickup_datetime": null,
  "delivery_datetime": null,
  "equipment_type": null,
  "mode": null,
  "rate": null,
  "currency": null,
  "weight": null,
  "carrier_name": null
}
"""

def run_extraction(vector_store):
    docs, display_scores, passed_threshold, _ = retrieve_with_scores(
        EXTRACTION_QUESTION, vector_store
    )

    # For extraction, use a lower threshold and fetch more chunks
    all_docs, _, _, _ = retrieve_with_scores("shipment shipper consignee rate carrier", vector_store)
    docs = list({d.page_content: d for d in docs + all_docs}.values())

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": EXTRACTION_QUESTION},
        return_only_outputs=True
    )

    raw = response["output_text"].strip()

    # Strip markdown fences if model ignores instructions
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    # Find JSON block
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        parsed = json.loads(raw)
        return parsed, True
    except json.JSONDecodeError:
        return raw, False


# ─────────────────────────────────────────────
# SIDEBAR — UPLOAD
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🚚 TMS Doc Intelligence")
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown("---")

        uploaded_files = st.file_uploader(
            "Upload Document(s)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, TXT"
        )

        if st.button("⚙️  Process Documents"):
            if uploaded_files:
                with st.spinner("Extracting text and building index..."):
                    raw_text = extract_text_from_files(uploaded_files)
                    if not raw_text.strip():
                        st.error("No text could be extracted from the uploaded files.")
                    else:
                        chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = get_vector_store(chunks)
                        st.session_state.raw_text = raw_text
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.file_names = [f.name for f in uploaded_files]
                        # Clear history on new upload
                        st.session_state.chat_history = []
                        st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
            else:
                st.warning("Please upload at least one file.")

        # Show processing stats
        if st.session_state.get("vector_store"):
            st.markdown("---")
            st.markdown("**📊 Index Stats**")
            names = st.session_state.get("file_names", [])
            for n in names:
                st.markdown(f"- `{n}`")
            st.markdown(f"- **{st.session_state.get('chunk_count', '?')} chunks** indexed")

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem; color:#475569; font-family:'IBM Plex Mono',monospace; line-height:1.7;">
        <b style="color:#64748b;">Guardrails active:</b><br>
        · Retrieval similarity threshold<br>
        · LLM confidence scoring<br>
        · "Not found" passthrough<br>
        · Low-confidence answer refusal
        </div>
        """, unsafe_allow_html=True)

    return uploaded_files


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Init session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    render_sidebar()

    # ── HEADER ──
    st.markdown('<div class="hero-title">📦 TMS Doc Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload logistics documents · Ask questions · Extract structured data</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:24px"></div>', unsafe_allow_html=True)

    if not st.session_state.vector_store:
        st.info("👈 Upload and process a document from the sidebar to get started.")
        return

    # ── TABS ──
    tab_qa, tab_extract, tab_history = st.tabs(["💬  Ask Questions", "🗂️  Extract Data", "📜  History"])

    # ══════════════════════════════════════════
    # TAB 1 — Q&A
    # ══════════════════════════════════════════
    with tab_qa:
        st.markdown("#### Ask a question about your document")
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        with st.form("qa_form", clear_on_submit=True):
            user_question = st.text_input(
                "Your question",
                placeholder="e.g. What is the carrier rate? When is pickup scheduled?",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Ask →")

        if submitted and user_question.strip():
            with st.spinner("Retrieving and reasoning..."):
                handle_question(user_question.strip(), st.session_state.vector_store)

        # Quick example chips
        st.markdown("---")
        st.markdown("**Example questions:**")
        cols = st.columns(3)
        examples = [
            "What is the carrier rate?",
            "Who is the consignee?",
            "When is pickup scheduled?",
            "What equipment type is required?",
            "What is the shipper's name?",
            "What is the delivery date?",
        ]
        for i, ex in enumerate(examples):
            with cols[i % 3]:
                if st.button(ex, key=f"ex_{i}"):
                    with st.spinner("Retrieving and reasoning..."):
                        handle_question(ex, st.session_state.vector_store)

    # ══════════════════════════════════════════
    # TAB 2 — STRUCTURED EXTRACTION
    # ══════════════════════════════════════════
    with tab_extract:
        st.markdown("#### Structured Shipment Data Extraction")
        st.markdown(
            "Automatically extracts all standard TMS fields from your document. "
            "Missing fields are returned as `null`.",
        )
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        if st.button("🔍  Run Extraction"):
            with st.spinner("Extracting structured data..."):
                result, success = run_extraction(st.session_state.vector_store)

            if success:
                st.success("Extraction complete.")
                # Pretty JSON display
                st.json(result)

                # Visual field table
                st.markdown("---")
                st.markdown("**Field Summary**")
                fields = [
                    ("Shipment ID", result.get("shipment_id")),
                    ("Shipper", result.get("shipper")),
                    ("Consignee", result.get("consignee")),
                    ("Pickup Datetime", result.get("pickup_datetime")),
                    ("Delivery Datetime", result.get("delivery_datetime")),
                    ("Equipment Type", result.get("equipment_type")),
                    ("Mode", result.get("mode")),
                    ("Rate", result.get("rate")),
                    ("Currency", result.get("currency")),
                    ("Weight", result.get("weight")),
                    ("Carrier Name", result.get("carrier_name")),
                ]
                col1, col2 = st.columns(2)
                for idx, (field, value) in enumerate(fields):
                    col = col1 if idx % 2 == 0 else col2
                    status = "✅" if value else "⬜"
                    col.markdown(f"{status} **{field}:** `{value if value else 'null'}`")
            else:
                st.warning("Could not parse JSON from model response. Raw output:")
                st.code(result, language="text")

    # ══════════════════════════════════════════
    # TAB 3 — CHAT HISTORY
    # ══════════════════════════════════════════
    with tab_history:
        st.markdown("#### Conversation History")

        if not st.session_state.chat_history:
            st.info("No questions asked yet in this session.")
        else:
            if st.button("🗑️  Clear History"):
                st.session_state.chat_history = []
                st.rerun()

            for i, item in enumerate(st.session_state.chat_history):
                conf = item.get("confidence")
                badge = confidence_badge_html(conf) if conf is not None else ""
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-q">Q: {item['user']}</div>
                    <div class="history-a">A: {item['bot'][:300]}{'...' if len(item['bot']) > 300 else ''}</div>
                    <div style="margin-top:4px">{badge}</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()