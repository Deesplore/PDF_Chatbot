# Chatbot
# 🚚 TMS Doc Intelligence

An AI-powered logistics document assistant. Upload a shipment document,
ask questions about it in plain English, and extract structured data —
all grounded in your document with confidence scoring and guardrails.

---

## What it does

- Upload logistics documents (PDF, DOCX, TXT)
- Ask natural language questions and get grounded answers
- See a confidence score and source text with every answer
- Auto-extract 11 standard TMS fields as structured JSON
- Refuses to answer when information isn't in the document

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/tms-doc-intelligence.git
cd tms-doc-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.env` file in the root folder:
```
GOOGLE_API_KEY=your_key_here
```

### 4. Run the UI
```bash
streamlit run app.py
```

### 5. Run the API (separate terminal)
```bash
uvicorn main:app --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload document → returns session_id |
| POST | `/ask` | Ask a question using session_id |
| POST | `/extract` | Extract structured shipment data |

Full API usage guide is in `docs/API_DOCUMENTATION.txt`

---

## Project Structure

```
tms-doc-intelligence/
├── app.py                      # Streamlit UI
├── main.py                     # FastAPI backend
├── requirements.txt
├── .env.example
└── docs/
    ├── HOW_TO_USE.txt          # UI user guide
    └── API_DOCUMENTATION.txt   # API testing guide
```

---

## Guardrails

The system includes 3 layers to prevent hallucinated answers:

1. **Retrieval threshold** — refuses to answer if document match is too weak
2. **Not found passthrough** — surfaces "not found" cleanly instead of guessing
3. **Confidence gate** — withholds answers the model isn't confident about

---

## Known Limitations

- Scanned image PDFs are not supported (text-based only)
- Sessions reset when the server restarts
- Works best with standard logistics document formats

---

## Environment Variables

```
GOOGLE_API_KEY=your_google_api_key_here
```