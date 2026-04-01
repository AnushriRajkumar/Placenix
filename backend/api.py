import io, json, time, sqlite3, asyncio, requests
from typing import Dict, Any, List, Optional
import fitz, pytesseract, numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from torch import no_grad

# ---------------- CONFIG ----------------
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_MIN_CHARS = 40
OCR_MAX_PAGES = 3
OCR_DPI = 250

# Use original embedding model again
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "placenix.db"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"   # switchable to "llama2" if you want

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- MODEL ----------------
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
emb_model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# ---------------- DB INIT ----------------
def db_conn(): return sqlite3.connect(DB_PATH, check_same_thread=False)
with db_conn() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, score REAL, data TEXT, ts TEXT
    )''')
    conn.commit()

# ---------------- HELPERS ----------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text
    if len(text.strip()) >= OCR_MIN_CHARS:
        return text.strip()
    # OCR fallback
    ocr_texts = []
    for i, page in enumerate(doc):
        if i >= OCR_MAX_PAGES:
            break
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        ocr_texts.append(pytesseract.image_to_string(img, lang="eng"))
    return "\n".join(ocr_texts).strip()

def embed_text(text: str) -> np.ndarray:
    txt = " ".join((text or "").split()[:1000])  # truncate for safety
    inputs = tokenizer(txt, return_tensors="pt", truncation=True, padding=True)
    with no_grad():
        outputs = emb_model(**inputs)
    return outputs.last_hidden_state.detach().numpy().mean(axis=1)

def call_ollama(prompt: str) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=40
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        return "⚠️ Ollama returned error."
    except Exception as e:
        print("Ollama error:", e)
        return "🔥 Quick Tips:\n1) Add measurable results\n2) Highlight technical stack\n3) Include project links"

async def save_analysis(user_id: str, score: float, structured: List[Dict[str, Any]], tips: str):
    try:
        with db_conn() as conn:
            conn.execute(
                'INSERT INTO analyses(user_id,score,data,ts) VALUES(?,?,?,?)',
                (user_id, float(score), json.dumps({"structured": structured, "tips": tips}), time.strftime('%Y-%m-%dT%H:%M:%S'))
            )
            conn.commit()
    except Exception as e:
        print("save_analysis error:", e)

# ---------------- ENDPOINTS ----------------
@app.post("/upload_resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form("Enter job description"),
    user_id: Optional[str] = Form(default="default")
) -> Dict[str, Any]:
    pdf_bytes = await file.read()
    resume_text = await asyncio.to_thread(extract_text_from_pdf, pdf_bytes)

    if not resume_text.strip():
        return {
            "structured_resume": [],
            "similarity_score": 0.0,
            "potential_promotions": ["Unknown"],
            "tips": "Resume unreadable.",
            "resume_text": ""
        }

    # Skills and roles
    skills = ["Python","C++","Java","SQL","Machine Learning","Deep Learning","TensorFlow","Pandas","Data Science","HTML","CSS","JavaScript","AWS","Azure"]
    detected = [s for s in skills if s.lower() in resume_text.lower()]
    role_map = {"Python":"Data Analyst","SQL":"Database Engineer","Machine Learning":"ML Engineer","Deep Learning":"AI Engineer","AWS":"Cloud Engineer","HTML":"Frontend Dev","JavaScript":"Fullstack Dev"}
    potential = list({role_map[s] for s in detected if s in role_map}) or ["Entry-level IT Support"]

    structured = [{"entity_group":"Skill","word":s,"score":0.9} for s in detected] + \
                 [{"entity_group":"Potential Role","word":r,"score":0.95} for r in potential]

    # Similarity
    try:
        resume_vec = await asyncio.to_thread(embed_text, resume_text)
        job_vec = await asyncio.to_thread(embed_text, job_description)
        sim = float(cosine_similarity(resume_vec, job_vec)[0][0])
    except Exception as e:
        sim = 0.0

    # Ollama tips
    prompt = f"""You are PhoenixMentor 🔥, a placement mentor.
Skills detected: {', '.join(detected) or 'None'}
Target job: {job_description}
Give 3 concise improvement tips."""
    tips = call_ollama(prompt)

    asyncio.create_task(save_analysis(user_id, sim, structured, tips))
    return {
        "structured_resume": structured,
        "similarity_score": sim,
        "potential_promotions": potential,
        "tips": tips,
        "resume_text": resume_text[:1500]
    }

@app.post("/chat")
async def chat(payload: Dict[str, Any]) -> Dict[str, str]:
    q = payload.get("q", "")
    if not q:
        return {"response": "Ask me something about your resume or job prep."}
    prompt = f"You are PhoenixMentor 🔥. The user asks: {q}. Reply with 3 helpful tips."
    return {"response": call_ollama(prompt)}

@app.get("/progress")
def progress(user_id: str = "default") -> Dict[str, List[Dict[str, Any]]]:
    with db_conn() as conn:
        rows = conn.execute('SELECT ts,score FROM analyses WHERE user_id=? ORDER BY ts', (user_id,)).fetchall()
    return {"series": [{"ts": ts, "score": float(score)} for ts, score in rows]}