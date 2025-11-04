import torch
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import pandas as pd
import json

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

def extract_text(path):
    reader = PdfReader(path)
    return " ".join([p.extract_text() or "" for p in reader.pages])

resume_path = "resumes/your_resume.pdf"
resume_text = extract_text(resume_path)

skills_df = pd.read_csv("skills_master.csv")
skill_texts = skills_df['skill_name'].tolist()

resume_emb = model.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
skill_embs = model.encode(skill_texts, convert_to_tensor=True, normalize_embeddings=True)

scores = util.cos_sim(resume_emb, skill_embs)[0]
topk = torch.topk(scores, k=20)
top_idx = topk.indices.cpu().numpy()
top_vals = topk.values.cpu().numpy()

threshold = 0.70
filtered = [
    {"skill": skill_texts[i], "score": float(top_vals[j])}
    for j, i in enumerate(top_idx) if top_vals[j] >= threshold
]

profile = {
    "user_id": "resume-test-001",
    "skills": [x["skill"] for x in filtered],
    "avg_confidence": float(sum([x["score"] for x in filtered]) / len(filtered))
}
print(json.dumps(profile, indent=2))
