from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
import torch
import torch.nn.functional as F

# ------------------- Resume NER -------------------
ner_model_id = "yashpwr/resume-ner-bert-v2"

ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_id)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_id)

ner_pipeline = pipeline(
    "ner",
    model=ner_model,
    tokenizer=ner_tokenizer,
    aggregation_strategy="simple"
)

# ------------------- Embeddings -------------------
emb_model_id = "distilbert-base-uncased"  # for similarity
emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_id)
emb_model = AutoModel.from_pretrained(emb_model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emb_model.to(device)

def get_embedding(text):
    inputs = emb_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2).item()

def compare_resume_with_job(resume_text, job_desc):
    resume_vec = get_embedding(resume_text)
    job_vec = get_embedding(job_desc)
    return cosine_similarity(resume_vec, job_vec)

# ------------------- Example -------------------
resume = "John Doe has 5 years experience in Python, Data Science, and Machine Learning."
job = "Looking for a Data Scientist skilled in Python and ML."

# NER Extraction
def analyze_resume(resume_text, job_desc):
    # Run NER
    ner_output = ner_pipeline(resume_text)

    # Run Similarity
    similarity_score = compare_resume_with_job(resume_text, job_desc)

    # Merge into one response
    result = {
        "structured_resume": ner_output,
        "similarity_score": round(similarity_score, 2)
    }
    return result


if __name__ == "__main__":
    resume = "John Doe has 5 years experience in Python, Data Science, and Machine Learning."
    job = "Looking for a Data Scientist skilled in Python and ML."

    final_output = analyze_resume(resume, job)
    print(final_output)  # frontend will later consume this
