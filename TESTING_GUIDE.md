# Quick Start Guide - Testing New Features

This guide walks you through testing the four newly implemented features.

## Prerequisites

- Backend running on `http://127.0.0.1:8001`
- Frontend running on `http://localhost:3000`
- Python 3.13+ environment with dependencies installed

## Feature 1: UI Toggles for Generative AI Controls

### Steps
1. Open the application at `http://localhost:3000`
2. Look for the **"Generative AI Controls"** panel in the input section
3. Toggle the following controls:
   - **Enable RAG**: Toggle to use Retrieval-Augmented Generation
   - **Tone**: Select from "Friendly", "Formal", or "Motivational"
   - **Response Length**: Choose "Short" or "Detailed"
   - **Show Draft & Final**: Toggle to see both draft and final versions
   - **Use LoRA**: Toggle to enable fine-tuned model (if available)

4. Fill in the form and click **"Analyze Career Path"**
5. Observe that the backend receives your control preferences

### Expected Result
The controls visually indicate your selections and are sent to the backend API.

---

## Feature 2: Real Agent Implementation Wiring

### Steps to Test Real Agents
1. Ensure you have API keys set up:
   - Set `GEMINI_API_KEY` in `.env.local` (frontend) or system environment
   - Or set `OPENAI_API_KEY` in system environment

2. Check backend health to verify agents are available:
   ```bash
   curl http://127.0.0.1:8001/api/health
   ```

3. You should see:
   ```json
   {
     "ok": true,
     "mode": "production",
     "agentsAvailable": true
   }
   ```

### Testing Without API Keys (Demo Mode)
1. Keep API keys unset
2. Check health endpoint - you'll see:
   ```json
   {
     "ok": true,
     "mode": "demo",
     "agentsAvailable": false
   }
   ```

3. The system automatically falls back to canned responses

### Testing Each Endpoint
```bash
# Parse resume
curl -X POST http://127.0.0.1:8001/api/parse \
  -H "Content-Type: application/json" \
  -d '{"resumeText": "5 years Python developer", ...}'

# Match roles
curl -X POST http://127.0.0.1:8001/api/match \
  -H "Content-Type: application/json" \
  -d '{"resumeText": "...", "targetRole": "Senior Backend Engineer", ...}'

# Retrieve RAG context
curl -X POST http://127.0.0.1:8001/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning roadmap"}'

# Full analysis with all controls
curl -X POST http://127.0.0.1:8001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resumeText": "10 years software engineer...",
    "targetRole": "ML Engineer",
    "weeklyHours": 10,
    "ragEnabled": true,
    "tone": "friendly",
    "length": "detailed",
    "showDraftFinal": true,
    "useLora": false
  }'
```

### Expected Result
- With API keys: Real agent responses with intelligent analysis
- Without API keys: Demo/canned responses (still valid for testing)

---

## Feature 3: Database for Psychometric Scores & Ratings

### Steps to Store & Retrieve Ratings

1. **Submit a Psychometric Score**
   ```bash
   curl -X POST http://127.0.0.1:8001/api/psychometric/score \
     -H "Content-Type: application/json" \
     -d '{
       "responses": {"1": "theory", "2": "structure", "3": "docs"},
       "user_id": "user-123"
     }'
   ```
   
   This automatically saves to database with psychometric profile.

2. **Submit a Roadmap Rating**
   ```bash
   curl -X POST http://127.0.0.1:8001/api/ratings/store \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user-123",
       "roadmap_id": "roadmap-abc",
       "grader_id": "grader-001",
       "usefulness": 5,
       "clarity": 4,
       "factuality": 5,
       "actionability": 4,
       "overall_rating": 4.5,
       "comments": "Excellent roadmap, very actionable"
     }'
   ```

3. **Retrieve All Ratings for a Roadmap**
   ```bash
   curl http://127.0.0.1:8001/api/ratings/list?roadmap_id=roadmap-abc
   ```

4. **Get Rating Summary (Mean ± StdDev)**
   ```bash
   curl http://127.0.0.1:8001/api/ratings/summary?roadmap_id=roadmap-abc
   ```

### Frontend Rating UI
1. After viewing a career roadmap, scroll to **"Rate This Roadmap"** section
2. Click a 5-star rating (1-5)
3. Optionally add comments
4. Click submit
5. You'll see a confirmation message
6. Check the database: `backend/careerpath.db`

### Verify Database

```bash
# On Windows, using Python
cd c:\Users\HP\Downloads\careerpath-ai\backend
python -c "
import sqlite3
conn = sqlite3.connect('careerpath.db')
c = conn.cursor()
print('Tables:', c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())
print('Ratings:', c.execute('SELECT * FROM roadmap_ratings LIMIT 5').fetchall())
"
```

### Expected Result
- Ratings stored in SQLite database
- Summary statistics computed (mean, stddev, count)
- Psychometric profiles linked to users

---

## Feature 4: LoRA Fine-tuning Pipeline

### Check LoRA Availability
```bash
curl http://127.0.0.1:8001/api/lora/status
```

Response example:
```json
{
  "lora_available": true,
  "gpu_available": false,
  "gpu_name": null,
  "recommendation": "LoRA training with GPU can be 10-100x faster"
}
```

### Configure LoRA Training
```bash
curl -X POST http://127.0.0.1:8001/api/lora/train \
  -H "Content-Type: application/json" \
  -d '{
    "num_samples": 50,
    "num_epochs": 3,
    "batch_size": 8,
    "model_name": "distilgpt2"
  }'
```

Response example:
```json
{
  "success": true,
  "message": "LoRA training configured successfully",
  "config": {
    "model": "distilgpt2",
    "num_samples": 50,
    "train_size": 45,
    "val_size": 5,
    "num_epochs": 3,
    "batch_size": 8,
    "gpu_available": false
  },
  "time_estimate": {
    "per_epoch_hours": 0.025,
    "total_3_epochs_hours": 0.075
  },
  "status": "gpu_recommended"
}
```

### Frontend LoRA Status
1. View any career roadmap
2. Scroll to **"LoRA Model Fine-tuning"** section
3. See:
   - LoRA availability (installed Y/N)
   - GPU availability (CUDA Y/N)
   - GPU name (if available)
   - Training recommendation

### To Actually Train (Requires GPU)
If you have a GPU (NVIDIA CUDA):

1. Install dependencies:
   ```bash
   pip install transformers peft torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. Create training script:
   ```python
   from backend.llm.lora_dataset import generate_qa_pairs
   from backend.llm.lora_train import LoRATrainer

   qa_pairs = generate_qa_pairs(n=50, seed=42)
   trainer = LoRATrainer(model_name="distilgpt2")
   trainer.load_model()
   
   train_texts, val_texts = trainer.prepare_dataset(qa_pairs)
   save_path = trainer.train(
       train_texts, 
       val_texts, 
       num_epochs=3, 
       batch_size=8
   )
   print(f"Model saved to {save_path}")
   ```

3. Run the script:
   ```bash
   python train_lora_local.py
   ```

### Expected Result
- LoRA configuration validated
- Time estimates provided
- GPU detection working
- Training infrastructure ready (actual training on GPU optional)

---

## Complete Integration Test

### Full User Flow
1. **Input Phase**
   - Use UI controls to set preferences
   - Fill resume, role, hours
   - Enable RAG, set tone to "friendly", length to "detailed"
   - Enable LoRA
   - Click analyze

2. **Processing Phase**
   - Backend receives all controls
   - Agents activate (or demo mode)
   - RAG retrieves relevant resources
   - LLM generates roadmap with specified tone/length

3. **Output Phase**
   - View roadmap in ResultsDashboard
   - See draft & final versions (if enabled)
   - See LoRA status panel

4. **Evaluation Phase**
   - Rate the roadmap (1-5 stars)
   - Add comments
   - Submit rating
   - Check database: `curl http://127.0.0.1:8001/api/ratings/list?roadmap_id=...`

5. **LoRA Training Phase** (optional)
   - Check GPU availability: `curl http://127.0.0.1:8001/api/lora/status`
   - Configure training: `curl -X POST http://127.0.0.1:8001/api/lora/train ...`
   - Train model (if GPU available)
   - Use fine-tuned model for future generations

---

## Troubleshooting

### Backend not starting
```bash
# Check Python path
.\.venv\Scripts\python.exe -c "import backend; print('OK')"

# Check uvicorn
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

### API returns 500 error
```bash
# Check logs in terminal running uvicorn
# Look for error messages
# Ensure API keys are set if using production agents
```

### Database errors
```bash
# Delete database and reinitialize
rm backend/careerpath.db
python -c "from backend.schema import init_db; init_db()"
```

### LoRA not available
```bash
# Install missing dependencies
pip install transformers peft torch
```

---

## Summary of Test Cases

| Feature | Test | Command | Expected |
|---------|------|---------|----------|
| **UI Controls** | Toggle RAG | Browser UI | Controls visible & functional |
| | Change tone | Browser UI | Options selectable |
| | Submit form | Browser UI | Controls sent to backend |
| **Agents** | Check health | `curl /api/health` | agentsAvailable: true/false |
| | Parse resume | `curl -X POST /api/parse` | Real or demo response |
| | Match roles | `curl -X POST /api/match` | Ranked matches |
| | Analyze full | `curl -X POST /api/analyze` | Complete roadmap |
| **Database** | Store rating | `curl -X POST /api/ratings/store` | success: true |
| | List ratings | `curl /api/ratings/list` | Array of ratings |
| | Summary | `curl /api/ratings/summary` | Mean ± StdDev |
| **LoRA** | Check status | `curl /api/lora/status` | Status object |
| | Configure train | `curl -X POST /api/lora/train` | Config & estimates |
| | Frontend panel | Browser UI | Status displayed |

---

Good luck testing! 🚀
