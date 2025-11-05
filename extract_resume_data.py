import json
from openai import OpenAI
from pathlib import Path
import docx
import PyPDF2

# Initialize client
client = OpenAI(api_key="YOUR_API_KEY_HERE")

def read_resume(file_path: str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif path.suffix.lower() == ".docx":
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return Path(file_path).read_text()

def extract_resume_json(resume_text: str) -> dict:
    prompt = f"""
    You are a structured information extraction model.
    From the following resume text, extract details in this JSON format:

    {{
      "user_id": "auto-generate-short-id",
      "name": "",
      "headline": "",
      "years_experience": 0,
      "education": [],
      "desired_roles": [],
      "skills": [],
      "location_keywords": [],
      "states": [],
      "remote_ok": false,
      "licensed_fields": [],
      "state_licenses": [],
      "desired_salary": {{"min": null, "max": null, "currency": "USD"}},
      "work_type_pref": [],
      "company_size_pref": [],
      "company_tags_pref": [],
      "geo_pref": {{"lat": null, "lon": null, "radius_km": null}},
      "notes": ""
    }}

    Make reasonable inferences (e.g., if they mention “open to remote work”, set remote_ok=true).
    Resume text:
    {resume_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured JSON data from resumes accurately."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    try:
        content = response.choices[0].message.content.strip()
        start = content.find('{')
        end = content.rfind('}') + 1
        data = json.loads(content[start:end])
        return data
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Raw output:", content)
        return {}

def main():
    file_path = input("Enter path to resume file: ")
    text = read_resume(file_path)
    data = extract_resume_json(text)
    output_path = Path(file_path).stem + "_parsed.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Extracted data saved to {output_path}")

if __name__ == "__main__":
    main()
