# Placenix – AI Career Guidance Platform

An AI-powered system that analyzes resumes, matches them with job descriptions, and provides personalized improvement feedback.

---

## Features

- Resume parsing using PDF + OCR  
- Transformer-based resume–job matching  
- Similarity scoring using cosine similarity  
- AI-generated feedback via chatbot  
- Progress tracking dashboard  

---

## Tech Stack

- Backend: FastAPI  
- Frontend: HTML, CSS, JavaScript  
- AI/ML: Sentence Transformers (MiniLM)  
- OCR: PyMuPDF, Tesseract  
- Database: SQLite  
- Chatbot: Ollama  

---

## How It Works

1. User uploads a resume  
2. Resume is parsed and converted into embeddings  
3. Compared with job description embeddings  
4. Similarity score is generated  
5. AI chatbot provides improvement suggestions  
6. Results are displayed on dashboard  

---

## Setup Instructions

1. Install dependencies:  
   pip install -r requirements.txt  

2. Run backend:  
   uvicorn api:app --reload  

3. Open frontend:  
   Open `index.html` in browser  

---

## Future Improvements

- Multi-job comparison  
- Skill gap analysis  
- Resume auto-enhancement  
- Job portal integration  
- Authentication system  
#uvicorn backend.api:app --reload --port 8001
#python -m http.server 8000
#ollama pull llama3.2ma:3b
#ollama serve"# Placenix" 
