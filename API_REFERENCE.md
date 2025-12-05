# CareerPath AI - Complete API Reference & Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React/TypeScript)                   │
│                        Port: 3000                                    │
├─────────────────────────────────────────────────────────────────────┤
│  • InputSection.tsx (Resume, Role, Controls: RAG, Tone, Length)     │
│  • ResultsDashboard.tsx (Roadmap Display, Rating, LoRA Status)      │
│  • Services: geminiService.ts, backendService.ts                    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ HTTP/JSON
                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 BACKEND API (FastAPI/Python)                         │
│                 Port: 8001                                           │
├─────────────────────────────────────────────────────────────────────┤
│  Health & Status (2 endpoints)                                      │
│  • GET /api/health                                                  │
│  • GET /api/lora/status                                             │
│                                                                     │
│  Career Analysis (6 endpoints)                                      │
│  • POST /api/parse          ┐                                       │
│  • POST /api/match          │ ─→ Agent Pipeline                    │
│  • POST /api/retrieve       │    (conditional)                      │
│  • POST /api/analyze        │                                       │
│  • POST /api/recommend      │                                       │
│  • POST /api/verify-links   ┘                                       │
│                                                                     │
│  Psychometric Testing (2 endpoints)                                 │
│  • GET /api/psychometric/questions                                  │
│  • POST /api/psychometric/score  ──→ Database Save                 │
│                                                                     │
│  Evaluation & Ratings (3 endpoints)                                │
│  • POST /api/ratings/store                                          │
│  • GET /api/ratings/list                                            │
│  • GET /api/ratings/summary                                         │
│                                                                     │
│  Synthetic Data & ML (3 endpoints)                                 │
│  • POST /api/synthetic/resumes                                      │
│  • POST /api/lora/train                                             │
│  • GET /api/lora/status                                             │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ↓                        ↓                        ↓
    ┌──────────────┐        ┌─────────────┐        ┌──────────────┐
    │   Agents     │        │  Database   │        │  LLM Models  │
    │              │        │  (SQLite)   │        │              │
    │ • parse_*    │        │             │        │ • Gemini API │
    │ • match_*    │        │ Tables:     │        │ • OpenAI API │
    │ • retrieve   │        │ • Psycho-   │        │              │
    │ • gen_*      │        │   metric    │        │ Optional:    │
    │ • verify_*   │        │   Scores    │        │ • LoRA       │
    │              │        │ • Roadmap   │        │ • Fine-tune  │
    │ Fallback:    │        │   Ratings   │        │              │
    │ • Demo mode  │        └─────────────┘        └──────────────┘
    │ • Canned     │
    │   responses  │
    └──────────────┘
```

---

## Complete Endpoint Reference

### 1. Health & Status Endpoints

#### GET `/api/health`
Check backend status and agent availability.

**Response:**
```json
{
  "ok": true,
  "mode": "production",
  "agentsAvailable": true
}
```

**Status Values:**
- `mode: "production"` - Real agents available
- `mode: "demo"` - Using canned responses
- `agentsAvailable: true` - Agents loaded successfully

---

#### GET `/api/lora/status`
Check LoRA training availability and GPU status.

**Response:**
```json
{
  "lora_available": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA A100",
  "recommendation": "GPU available for optimal training"
}
```

**Notes:**
- GPU detection via CUDA
- Helps users decide whether to initiate training

---

### 2. Career Analysis Endpoints

#### POST `/api/parse`
Extract information from resume text.

**Request:**
```json
{
  "resumeText": "10 years software engineer with Python, SQL, React...",
  "targetRole": "Senior Backend Engineer",
  "weeklyHours": 10,
  "ragEnabled": true,
  "tone": "friendly",
  "length": "detailed",
  "showDraftFinal": false,
  "useLora": false
}
```

**Response (Real Agent):**
```json
{
  "success": true,
  "name": "John Doe",
  "skills": ["Python", "SQL", "React", "Docker"],
  "experience": ["10 years as Software Engineer", "5 years as Tech Lead"],
  "education": ["B.S. Computer Science"],
  "summary": "Experienced backend engineer with leadership experience"
}
```

**Response (Demo):**
```json
{
  "success": true,
  "name": "John Doe",
  "skills": ["Python", "SQL", "JavaScript", "React"],
  "experience": ["3 years as developer"],
  "education": ["B.S. CS"],
  "summary": "Experienced developer"
}
```

---

#### POST `/api/match`
Match resume skills against available roles.

**Request:**
```json
{
  "resumeText": "...",
  "targetRole": "Data Scientist",
  "weeklyHours": 10,
  "ragEnabled": true,
  "tone": "formal",
  "length": "short",
  "showDraftFinal": true,
  "useLora": false
}
```

**Response:**
```json
{
  "matches": [
    {
      "role": "Machine Learning Engineer",
      "matchScore": 85,
      "reason": "Matched 85% of required skills"
    },
    {
      "role": "Data Analyst",
      "matchScore": 72,
      "reason": "Matched 72% of required skills"
    }
  ]
}
```

---

#### POST `/api/retrieve`
Retrieve learning resources using RAG.

**Request:**
```json
{
  "query": "machine learning fundamentals for backend engineers"
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc-001",
      "text": "Machine Learning Fundamentals...",
      "score": 0.92
    },
    {
      "id": "doc-002",
      "text": "Deep Learning with Python...",
      "score": 0.88
    }
  ]
}
```

**Notes:**
- Vector similarity search if transformers available
- Falls back to substring matching in corpus
- Score represents relevance (0.0-1.0)

---

#### POST `/api/analyze`
Full career analysis with all features.

**Request:**
```json
{
  "resumeText": "5 years Python backend engineer...",
  "targetRole": "ML Engineer",
  "weeklyHours": 10,
  "psychometricProfile": {
    "archetype": "Strategic Architect",
    "learningStyle": "Practical"
  },
  "ragEnabled": true,
  "tone": "motivational",
  "length": "detailed",
  "showDraftFinal": true,
  "useLora": false
}
```

**Response:**
```json
{
  "matchScore": 78,
  "matchSummary": "Career roadmap for ML Engineer",
  "resumeSummary": "5 years Python backend engineer...",
  "currentSkills": ["Python", "SQL", "Docker"],
  "missingSkills": ["TensorFlow", "Deep Learning"],
  "roadmap": [
    {
      "phase": 1,
      "name": "Fundamentals",
      "weeks": [...]
    }
  ],
  "immediateActions": [
    "Learn TensorFlow basics",
    "Complete 1-2 ML projects",
    "Join ML community"
  ],
  "interviewTips": [
    "Be ready to explain your ML projects",
    "Study common algorithms"
  ],
  "final_roadmap": "Complete detailed roadmap...",
  "draft_roadmap": "Rough draft version...",
  "critique": "Feedback on draft..."
}
```

**Control Mappings:**
- `tone`: "friendly" | "formal" | "motivational" → influences language style
- `length`: "short" | "detailed" → affects roadmap granularity
- `showDraftFinal`: true/false → includes draft and critique
- `ragEnabled`: true/false → includes retrieved resources
- `useLora`: true/false → uses fine-tuned model if available

---

#### POST `/api/recommend`
Get role recommendations based on skills.

**Request:**
```json
{
  "resumeText": "Python developer with 5 years experience...",
  "interests": "machine learning, data science",
  "workStyle": "collaborative"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "role": "ML Engineer",
      "matchScore": 85,
      "reason": "Strong Python background, ML interest"
    },
    {
      "role": "Data Scientist",
      "matchScore": 80,
      "reason": "Experience with Python, analytical mindset"
    },
    {
      "role": "Backend Engineer",
      "matchScore": 75,
      "reason": "5 years Python experience"
    }
  ]
}
```

---

#### POST `/api/verify-links`
Verify that resources/links are accessible.

**Request:**
```json
{
  "links": [
    "https://www.tensorflow.org",
    "https://pytorch.org",
    "https://scikit-learn.org"
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "url": "https://www.tensorflow.org",
      "ok": true,
      "status": 200
    },
    {
      "url": "https://pytorch.org",
      "ok": true,
      "status": 200
    },
    {
      "url": "https://scikit-learn.org",
      "ok": true,
      "status": 200
    }
  ]
}
```

---

### 3. Psychometric Testing Endpoints

#### GET `/api/psychometric/questions`
Retrieve psychometric assessment questions.

**Response:**
```json
{
  "questions": [
    {
      "id": 1,
      "text": "When facing a completely new problem, what is your first instinct?",
      "options": ["theory", "practical", "visual", "social"]
    },
    {
      "id": 2,
      "text": "In a team setting, which role do you naturally gravitate towards?",
      "options": ["structure", "deep_work", "connector", "driver"]
    },
    {
      "id": 3,
      "text": "How do you prefer to learn a new technology?",
      "options": ["docs", "video", "project"]
    },
    {
      "id": 4,
      "text": "What motivates you most in a career?",
      "options": ["mastery", "creation", "logic"]
    }
  ]
}
```

---

#### POST `/api/psychometric/score`
Score psychometric responses and auto-save to database.

**Request:**
```json
{
  "responses": {
    "1": "practical",
    "2": "structure",
    "3": "project",
    "4": "mastery"
  },
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "profile": {
    "archetype": "Strategic Architect",
    "learningStyle": "Practical",
    "workStyle": "Collaborative",
    "traits": ["Analytical", "Pragmatic", "Strategic"],
    "description": "Strategic Architect who learns best via Practical methods and thrives in collaborative environments."
  }
}
```

**Side Effects:**
- Automatically saves to `psychometric_scores` table
- Linked to `user_id` for tracking

---

### 4. Ratings & Evaluation Endpoints

#### POST `/api/ratings/store`
Store a grader's rating for a roadmap.

**Request:**
```json
{
  "user_id": "user-123",
  "roadmap_id": "roadmap-abc-789",
  "grader_id": "grader-001",
  "usefulness": 5,
  "clarity": 4,
  "factuality": 5,
  "actionability": 4,
  "overall_rating": 4.5,
  "comments": "Excellent and very actionable roadmap. Only minor clarity issues."
}
```

**Response:**
```json
{
  "success": true,
  "rating_id": 42,
  "message": "Rating stored successfully"
}
```

**or (Database unavailable):**
```json
{
  "success": false,
  "error": "Database not available. Ratings will not persist.",
  "rating": {...}
}
```

**Rating Dimensions:**
- `usefulness` (1-5): How useful is this roadmap?
- `clarity` (1-5): How clear are the instructions?
- `factuality` (1-5): How accurate is the information?
- `actionability` (1-5): How actionable are the steps?
- `overall_rating` (1-5): Overall score
- `comments` (string): Optional feedback

---

#### GET `/api/ratings/list?roadmap_id=roadmap-abc-789`
Get all ratings for a specific roadmap.

**Response:**
```json
{
  "roadmap_id": "roadmap-abc-789",
  "total_ratings": 3,
  "ratings": [
    {
      "id": 1,
      "grader_id": "grader-001",
      "usefulness": 5,
      "clarity": 4,
      "factuality": 5,
      "actionability": 4,
      "overall_rating": 4.5,
      "comments": "Excellent roadmap",
      "created_at": "2024-01-15T10:30:00"
    },
    {
      "id": 2,
      "grader_id": "grader-002",
      "usefulness": 4,
      "clarity": 5,
      "factuality": 4,
      "actionability": 5,
      "overall_rating": 4.5,
      "comments": "Very clear and actionable",
      "created_at": "2024-01-15T11:45:00"
    },
    {
      "id": 3,
      "grader_id": "grader-003",
      "usefulness": 5,
      "clarity": 4,
      "factuality": 5,
      "actionability": 4,
      "overall_rating": 4.5,
      "comments": "Practical and well-structured",
      "created_at": "2024-01-15T13:20:00"
    }
  ]
}
```

---

#### GET `/api/ratings/summary?roadmap_id=roadmap-abc-789`
Compute aggregate statistics for a roadmap's ratings.

**Response:**
```json
{
  "roadmap_id": "roadmap-abc-789",
  "summary": {
    "total_ratings": 3,
    "usefulness": {
      "mean": 4.67,
      "stddev": 0.47,
      "count": 3
    },
    "clarity": {
      "mean": 4.33,
      "stddev": 0.47,
      "count": 3
    },
    "factuality": {
      "mean": 4.67,
      "stddev": 0.47,
      "count": 3
    },
    "actionability": {
      "mean": 4.33,
      "stddev": 0.47,
      "count": 3
    },
    "overall": {
      "mean": 4.5,
      "stddev": 0.0,
      "count": 3
    }
  }
}
```

**Statistics:**
- `mean`: Average rating (1-5)
- `stddev`: Standard deviation (consistency)
- `count`: Number of ratings for this dimension

**Interpretation:**
- High mean, low stddev → Consistent excellent quality
- Low mean, high stddev → Mixed quality or polarized opinions
- High stddev → Graders disagree significantly

---

### 5. Synthetic Data & ML Endpoints

#### POST `/api/synthetic/resumes`
Generate synthetic resumes for training/evaluation.

**Request:**
```json
{
  "n": 10
}
```

**Response:**
```json
{
  "generated": [
    {
      "summary": "Synthetic resume 1: 3-year software engineer with Python and React...",
      "skills": "Python, React, SQL, Git, Docker",
      "projects": [
        "Built e-commerce platform with React and Python backend",
        "Developed CLI tool for data processing"
      ]
    },
    {
      "summary": "Synthetic resume 2: Junior data analyst with SQL and Python...",
      "skills": "Python, SQL, Excel, Tableau, Statistics",
      "projects": [
        "Built data dashboard for sales tracking",
        "Analyzed customer behavior patterns"
      ]
    }
    // ... 8 more
  ]
}
```

---

#### POST `/api/lora/train`
Configure and estimate LoRA fine-tuning training.

**Request:**
```json
{
  "num_samples": 50,
  "num_epochs": 3,
  "batch_size": 8,
  "model_name": "distilgpt2"
}
```

**Response (Success):**
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
    "gpu_available": true
  },
  "time_estimate": {
    "per_epoch_hours": 0.015,
    "total_3_epochs_hours": 0.045,
    "note": "Estimates are rough; actual times depend on batch size and hardware"
  },
  "status": "ready",
  "note": "Actual training requires GPU. Use: trainer.train(train_texts, val_texts, ...)"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "LoRA modules not available. Install with: pip install transformers peft torch"
}
```

**Parameters:**
- `num_samples`: Number of Q&A pairs to generate (10-1000+)
- `num_epochs`: Training epochs (1-10, typical 3)
- `batch_size`: Batch size (4, 8, 16, 32 depending on GPU)
- `model_name`: Base model (distilgpt2, gpt2, mistral-7b, etc.)

**Time Estimates (rough):**
| Config | CPU | GPU |
|--------|-----|-----|
| 50 samples, 1 epoch | 1-2 min | 1-5 sec |
| 500 samples, 3 epochs | 30-60 min | 1-2 min |
| 5000 samples, 3 epochs | 5-10 hrs | 5-15 min |

---

## Request/Response Pattern

All endpoints follow consistent patterns:

**Success Response (2xx):**
```json
{
  "success": true,
  "data": { ... },
  "message": "Optional success message"
}
```

**Error Response (4xx/5xx):**
```json
{
  "success": false,
  "error": "Descriptive error message",
  "details": "Optional detailed explanation"
}
```

---

## Control Parameters

Control parameters passed in requests affect generation behavior:

```python
{
  "ragEnabled": bool,           # Use RAG retrieval
  "tone": str,                  # "friendly" | "formal" | "motivational"
  "length": str,                # "short" | "detailed"
  "showDraftFinal": bool,       # Include draft and critique versions
  "useLora": bool               # Use fine-tuned model if available
}
```

**Backend Processing:**
```python
if AGENTS_AVAILABLE:
    # Pass controls to agents
    if request.ragEnabled:
        context = retrieve(query, k=6)
    
    response = generate_roadmap_chain(
        ...,
        tone=request.tone,
        length=request.length
    )
    
    if request.showDraftFinal:
        return {
            "draft": draft_version,
            "critique": critique,
            "final": final_version
        }
else:
    # Use demo response (ignores controls)
    return demo_cached_response
```

---

## Error Handling

### Common HTTP Status Codes

| Code | Scenario | Example |
|------|----------|---------|
| 200 | Success | Resource retrieved/created |
| 400 | Bad Request | Missing required field |
| 422 | Validation Error | Invalid data type |
| 500 | Server Error | Unhandled exception in agent |
| 503 | Service Unavailable | Database connection failed |

### Error Response Example

```json
{
  "detail": "Invalid input: resumeText must be at least 10 characters"
}
```

---

## Rate Limits & Performance

### Recommended Limits (Production)
- `/api/analyze`: 1 req/second per user (computationally expensive)
- `/api/parse`: 5 req/second per user (light parsing)
- `/api/ratings/*`: 10 req/second (database operations)
- `/api/lora/train`: 1 per hour (long-running background job)

### Response Times (Typical)
| Endpoint | With Agent | Demo |
|----------|-----------|------|
| `/api/parse` | 2-5s | 50ms |
| `/api/match` | 1-3s | 50ms |
| `/api/analyze` | 10-30s | 100ms |
| `/api/ratings/store` | 100-500ms | 50ms |
| `/api/lora/status` | 200ms | 200ms |

---

## Authentication (Future)

Currently no authentication. In production, add:

```python
from fastapi.security import HTTPBearer
from fastapi import Depends, HTTPException

security = HTTPBearer()

@app.post('/api/analyze')
async def api_analyze(req: AnalyzeRequest, credentials = Depends(security)):
    user_id = verify_token(credentials.credentials)
    # ... process request
```

---

## Summary

**15 Endpoints Total:**
- 2 Health/Status
- 6 Career Analysis
- 2 Psychometric
- 3 Ratings/Evaluation
- 2 Synthetic Data/ML

**All endpoints support:**
- Conditional real agents
- Graceful fallback to demo
- Control parameters (tone, length, RAG, LoRA)
- Error handling and detailed responses

**Database persists:**
- Psychometric test scores
- Roadmap ratings and statistics

**LoRA training:**
- Configurable parameters
- GPU detection and time estimation
- Production-ready infrastructure

Happy coding! 🚀
