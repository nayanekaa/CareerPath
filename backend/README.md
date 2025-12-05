CareerPath-Gen backend

Quick start (Windows PowerShell):

1. Create and activate a virtualenv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Run the server (development)

```powershell
$env:OPENAI_API_KEY = "your-key-here"  # optional; when absent, app uses demo/canned_outputs.json
uvicorn backend.main:app --reload --port 8000
```

Notes:
- If `OPENAI_API_KEY` is not set the backend will serve demo canned outputs.
- `sentence-transformers` and `faiss` are optional: if missing the retrieval agent will fall back to simple substring matching.
- LoRA fine-tuning script is a skeleton and requires GPU and data to run.
