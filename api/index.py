import os
import io
import json
import pdfplumber
import docx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Using a specific model that is guaranteed to stay on the free tier
hf_client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

def extract_text(file: UploadFile, content: bytes):
    try:
        if file.filename.endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        elif file.filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs])
        return content.decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error: {e}"

@app.post("/api/analyze")
async def analyze(resume: UploadFile = File(...), jd: str = Form(...)):
    try:
        content = await resume.read()
        resume_text = extract_text(resume, content)
        
        prompt = (
            f"Return ONLY JSON. No prose. "
            f"Format: {{\"score\": 85, \"missing_keywords\": [\"React\"], \"skills_gap\": {{\"Tech\": 80, \"Soft\": 60}}}}. "
            f"JD: {jd} Resume: {resume_text}"
        )

        try:
            # Primary: Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return response.parsed
        
        except Exception:
            print("Gemini Busy. Using Microsoft Phi-3 Backup...")
            # Backup: Phi-3 is tiny, fast, and great at following JSON instructions
            hf_res = hf_client.text_generation(
                model="microsoft/Phi-3-mini-4k-instruct",
                prompt=f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
                max_new_tokens=500
            )
            
            # Extract JSON safely
            start = hf_res.find('{')
            end = hf_res.rfind('}') + 1
            return json.loads(hf_res[start:end])

    except Exception as e:
        print(f"Critical System Error: {e}")
        return {
            "score": 45, # Default "Something went wrong" score
            "missing_keywords": ["API Busy - Please try again"], 
            "skills_gap": {"System": 10}
        }

@app.get("/")
def health(): return {"status": "Server Online"}