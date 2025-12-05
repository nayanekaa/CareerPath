from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import sys

# Ensure backend module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import agent modules; if they fail, we'll use demo responses
try:
    from backend.agents.parse_agent import parse_resume
    from backend.agents.match_agent import match_roles
    from backend.agents.retrieval_agent import retrieve
    # Import gen_agent as a module and guard optional exports
    import backend.agents.gen_agent as gen_agent_module
    generate_roadmap_chain = getattr(gen_agent_module, 'generate_roadmap_chain', None)
    # synthetic resume generator is optional
    generate_synthetic_resumes = getattr(gen_agent_module, 'generate_synthetic_resumes', None)
    from backend.agents.verify_agent import verify_links
    from backend.agents.psychometric_agent import score_responses
    AGENTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import agents: {e}. Using demo responses only.")
    AGENTS_AVAILABLE = False

# Import database modules
try:
    from backend.schema import init_db
    from backend.db import PsychometricDB, RoadmapRatingDB
    init_db()  # Initialize database on startup
    DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not initialize database: {e}. Ratings will not persist.")
    DB_AVAILABLE = False

DEMO_PATH = os.path.join(os.path.dirname(__file__), 'demo', 'canned_outputs.json')

app = FastAPI(title='CareerPath-Gen Backend')


class AnalyzeRequest(BaseModel):
    resumeText: str
    targetRole: str
    weeklyHours: int
    psychometricProfile: Optional[dict] = None
    ragEnabled: bool = True
    tone: str = 'friendly'
    length: str = 'detailed'
    showDraftFinal: bool = False
    useLora: bool = False


class RoadmapRatingRequest(BaseModel):
    user_id: str
    roadmap_id: str
    grader_id: str
    usefulness: Optional[int] = None
    clarity: Optional[int] = None
    factuality: Optional[int] = None
    actionability: Optional[int] = None
    overall_rating: Optional[int] = None
    comments: Optional[str] = None


class RoleRecRequest(BaseModel):
    resumeText: str
    interests: str
    workStyle: str


@app.get('/api/health')
def health():
    return {
        'ok': True,
        'status': 'backend running',
        'agentsAvailable': AGENTS_AVAILABLE,
        'mode': 'production' if AGENTS_AVAILABLE else 'demo'
    }


@app.post('/api/parse')
def api_parse(req: AnalyzeRequest):
    """Parse resume (demo mode)."""
    return {
        'summary': req.resumeText[:200],
        'skills': ['Python', 'SQL', 'JavaScript']
    }


@app.post('/api/match')
def api_match(req: AnalyzeRequest):
    """Match roles using real agent or demo."""
    if AGENTS_AVAILABLE:
        try:
            parsed = parse_resume(req.resumeText)
            matches = match_roles(parsed.get('skills', []), req.targetRole)
            return {
                'matches': [
                    {'role': m.get('role', 'Unknown'), 'matchScore': int(m.get('overlap_score', 0)*100), 'reason': f"Matched {m.get('overlap_score', 0):.0%} of skills"}
                    for m in matches
                ]
            }
        except Exception as e:
            print(f"Match agent error: {e}")

    # Demo/fallback response
    return {'matches': [
        {'role': 'Data Analyst', 'matchScore': 85, 'reason': 'Good fit'},
        {'role': 'Backend Engineer', 'matchScore': 72, 'reason': 'Some overlap'}
    ]}


@app.post('/api/retrieve')
def api_retrieve(q: dict):
    """Retrieve RAG context using real agent or demo."""
    query = q.get('query', '')

    if AGENTS_AVAILABLE:
        try:
            results = retrieve(query, k=6)
            return {'results': results}
        except Exception as e:
            print(f"Retrieve agent error: {e}")

    # Demo/fallback response
    return {'results': [
        {'id': 'r1', 'text': 'Python fundamentals and core concepts...', 'score': 0.9},
        {'id': 'r2', 'text': 'Data structures and algorithms...', 'score': 0.85}
    ]}


@app.post('/api/analyze')
def api_analyze(req: AnalyzeRequest):
    """Analyze career path using real agents or demo responses."""
    # Try to use real agents if available
    if AGENTS_AVAILABLE:
        try:
            parsed = parse_resume(req.resumeText)
            matches = match_roles(parsed.get('skills', []), req.targetRole)

            # Use RAG if enabled
            retrieved = []
            if req.ragEnabled:
                retrieved = retrieve(f"{req.targetRole} {' '.join(parsed.get('skills', []))}", k=6)

            # Generate roadmap with controls
            draft, critique, final = generate_roadmap_chain(
                parsed,
                req.targetRole,
                req.weeklyHours,
                retrieved,
                req.psychometricProfile,
                tone=req.tone,
                length=req.length
            )

            # Only return draft if showDraftFinal enabled, else just final
            result = {
                'matchScore': 75,
                'matchSummary': f'Career roadmap for {req.targetRole}',
                'resumeSummary': parsed.get('summary', req.resumeText[:100]),
                'currentSkills': parsed.get('skills', []),
                'missingSkills': [],
                'roadmap': [],
                'immediateActions': ['Start with fundamentals', 'Build projects', 'Network with professionals'],
                'interviewTips': ['Practice system design', 'Review company culture', 'Prepare behavioral stories'],
                'final_roadmap': final,
            }
            if req.showDraftFinal:
                result['draft_roadmap'] = draft
                result['critique'] = critique

            return result
        except Exception as e:
            print(f"Agent error: {e}")
            # Fall through to demo mode

    # Demo/fallback response
    if os.path.exists(DEMO_PATH):
        try:
            with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                canned = json.load(f)
            return canned.get('analysis', {})
        except Exception:
            pass

    return {
        'matchScore': 75,
        'matchSummary': f'Career roadmap for {req.targetRole}',
        'resumeSummary': req.resumeText[:100],
        'currentSkills': ['Python', 'SQL'],
        'missingSkills': [],
        'roadmap': [],
        'immediateActions': ['Learn target skills', 'Build portfolio', 'Network'],
        'interviewTips': ['Prepare technical questions', 'Research company', 'Practice interviews']
    }


@app.post('/api/recommend')
def api_recommend(req: RoleRecRequest):
    """Recommend roles using real agent or demo."""
    if AGENTS_AVAILABLE:
        try:
            parsed = parse_resume(req.resumeText) if req.resumeText else {}
            matches = match_roles(parsed.get('skills', []), '')
            return {
                'recommendations': [
                    {'role': m.get('role', 'Unknown'), 'matchScore': int(m.get('overlap_score', 0)*100), 'reason': f"Matched {m.get('overlap_score', 0):.0%} of skills"}
                    for m in matches[:5]
                ]
            }
        except Exception as e:
            print(f"Recommend agent error: {e}")

    # Demo/fallback response
    if os.path.exists(DEMO_PATH):
        try:
            with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                canned = json.load(f)
            return canned.get('recommendations', {'recommendations': []})
        except Exception:
            pass

    return {'recommendations': [
        {'role': 'Data Analyst', 'matchScore': 85, 'reason': 'Good skill overlap'},
        {'role': 'Machine Learning Engineer', 'matchScore': 78, 'reason': 'Strong technical background'}
    ]}


@app.post('/api/verify-links')
def api_verify(payload: dict):
    """Verify links using real agent or demo."""
    links = payload.get('links', [])

    if AGENTS_AVAILABLE:
        try:
            results = verify_links(links)
            return {'results': results}
        except Exception as e:
            print(f"Verify agent error: {e}")

    # Demo/fallback response
    return {'results': [{'url': link, 'ok': True, 'status': 200} for link in links]}


@app.get('/api/psychometric/questions')
def psychometric_questions():
    """Get psychometric questions."""
    return {
        'questions': [
            {'id': 1, 'text': 'When facing a completely new problem, what is your first instinct?', 'options': ['theory','practical','visual','social']},
            {'id': 2, 'text': 'In a team setting, which role do you naturally gravitate towards?', 'options': ['structure','deep_work','connector','driver']},
            {'id': 3, 'text': 'How do you prefer to learn a new technology?', 'options': ['docs','video','project']},
            {'id': 4, 'text': 'What motivates you most in a career?', 'options': ['mastery','creation','logic']},
        ]
    }


@app.post('/api/psychometric/score')
def psychometric_score(payload: dict):
    """Score psychometric responses using real agent or demo, and save to database."""
    responses = payload.get('responses', {})
    user_id = payload.get('user_id', 'anonymous')

    profile = None

    if AGENTS_AVAILABLE:
        try:
            profile = score_responses(responses)
        except Exception as e:
            print(f"Psychometric agent error: {e}")

    # Demo/fallback response
    if not profile:
        if os.path.exists(DEMO_PATH):
            try:
                with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                    canned = json.load(f)
                profile = canned.get('psychometric', {}).get('profile', {})
            except Exception:
                pass

        if not profile:
            profile = {
                'archetype': 'Strategic Architect',
                'learningStyle': 'Practical',
                'workStyle': 'Collaborative',
                'traits': ['Analytical', 'Pragmatic'],
                'description': 'Strategic Architect who learns best via Practical methods.'
            }

    # Save to database if available
    if DB_AVAILABLE:
        try:
            PsychometricDB.save_score(
                user_id=user_id,
                test_responses=responses,
                archetype=profile.get('archetype', ''),
                learning_style=profile.get('learningStyle', ''),
                work_style=profile.get('workStyle', ''),
                traits=profile.get('traits', []),
                description=profile.get('description', '')
            )
        except Exception as e:
            print(f"Database save error: {e}")

    return {'profile': profile}


@app.post('/api/synthetic/resumes')
def gen_synthetic(payload: dict):
    """Generate synthetic resumes using real agent or demo."""
    n = int(payload.get('n', 10))

    if AGENTS_AVAILABLE:
        try:
            resumes = generate_synthetic_resumes(n=n)
            return {'generated': resumes}
        except Exception as e:
            print(f"Synthetic agent error: {e}")

    # Demo/fallback response
    if os.path.exists(DEMO_PATH):
        try:
            with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                canned = json.load(f)
            return canned.get('synthetic', {'generated': []})
        except Exception:
            pass

    return {'generated': [
        {'summary': f'Synthetic resume {i+1}', 'skills': 'Python,SQL', 'projects': ['Project 1', 'Project 2']}
        for i in range(n)
    ]}


@app.post('/api/lora/train')
def lora_train(payload: dict):
    """
    Start LoRA fine-tuning on career-related Q&A dataset.

    This is a long-running task. In production, this should be async
    or queued to a background job service.
    """
    try:
        from backend.llm.lora_dataset import generate_qa_pairs
        from backend.llm.lora_train import LoRATrainer
    except ImportError:
        return {
            'success': False,
            'error': 'LoRA modules not available. Install with: pip install transformers peft torch'
        }

    # Get parameters
    num_samples = int(payload.get('num_samples', 50))
    num_epochs = int(payload.get('num_epochs', 3))
    batch_size = int(payload.get('batch_size', 8))
    model_name = payload.get('model_name', 'distilgpt2')

    try:
        # Generate dataset
        qa_pairs = generate_qa_pairs(n=num_samples, seed=42)

        # Initialize trainer
        trainer = LoRATrainer(model_name=model_name)

        # Load model with LoRA
        if not trainer.load_model():
            return {
                'success': False,
                'error': f'Failed to load model: {model_name}'
            }

        # Prepare dataset
        train_texts, val_texts = trainer.prepare_dataset(qa_pairs)

        # Estimate time
        from backend.llm.lora_train import estimate_training_time
        time_estimate = estimate_training_time(
            num_samples=len(train_texts),
            model_size='small',
            has_gpu=trainer.has_gpu
        )

        # For demo: don't actually train (too slow without GPU)
        # In production with GPU, uncomment the train call:
        # save_path = trainer.train(train_texts, val_texts, num_epochs=num_epochs, batch_size=batch_size)

        return {
            'success': True,
            'message': 'LoRA training configured successfully',
            'config': {
                'model': model_name,
                'num_samples': num_samples,
                'train_size': len(train_texts),
                'val_size': len(val_texts),
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'gpu_available': trainer.has_gpu,
            },
            'time_estimate': time_estimate,
            'status': 'ready' if trainer.has_gpu else 'gpu_recommended',
            'note': 'Actual training requires GPU. Use: trainer.train(train_texts, val_texts, ...)'
        }

    except Exception as e:
        print(f"LoRA training error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.get('/api/lora/status')
def lora_status():
    """Check LoRA training availability and GPU status."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except ImportError:
        gpu_available = False
        gpu_name = None

    try:
        from backend.llm.lora_train import LoRATrainer
        lora_available = True
    except ImportError:
        lora_available = False

    return {
        'lora_available': lora_available,
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'recommendation': 'LoRA training with GPU can be 10-100x faster' if not gpu_available else 'GPU available for optimal training'
    }


@app.post('/api/ratings/store')
def store_rating(req: RoadmapRatingRequest):
    """Store a grader's rating for a roadmap."""
    if not DB_AVAILABLE:
        return {
            'success': False,
            'error': 'Database not available. Ratings will not persist.',
            'rating': {
                'user_id': req.user_id,
                'roadmap_id': req.roadmap_id,
                'grader_id': req.grader_id
            }
        }

    try:
        rating = RoadmapRatingDB.save_rating(
            user_id=req.user_id,
            roadmap_id=req.roadmap_id,
            grader_id=req.grader_id,
            usefulness=req.usefulness,
            clarity=req.clarity,
            factuality=req.factuality,
            actionability=req.actionability,
            overall_rating=req.overall_rating,
            comments=req.comments
        )
        return {
            'success': True,
            'rating_id': rating.id,
            'message': 'Rating stored successfully'
        }
    except Exception as e:
        print(f"Rating storage error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.get('/api/ratings/list')
def list_ratings(roadmap_id: str):
    """Get all ratings for a specific roadmap."""
    if not DB_AVAILABLE:
        return {
            'roadmap_id': roadmap_id,
            'ratings': [],
            'warning': 'Database not available'
        }

    try:
        ratings = RoadmapRatingDB.get_ratings(roadmap_id)
        return {
            'roadmap_id': roadmap_id,
            'total_ratings': len(ratings),
            'ratings': [
                {
                    'id': r.id,
                    'grader_id': r.grader_id,
                    'usefulness': r.usefulness,
                    'clarity': r.clarity,
                    'factuality': r.factuality,
                    'actionability': r.actionability,
                    'overall_rating': r.overall_rating,
                    'comments': r.comments,
                    'created_at': r.created_at.isoformat() if r.created_at else None
                }
                for r in ratings
            ]
        }
    except Exception as e:
        print(f"Rating retrieval error: {e}")
        return {
            'roadmap_id': roadmap_id,
            'ratings': [],
            'error': str(e)
        }


@app.get('/api/ratings/summary')
def rating_summary(roadmap_id: str):
    """Compute summary statistics (mean ± stddev) for a roadmap's ratings."""
    if not DB_AVAILABLE:
        return {
            'roadmap_id': roadmap_id,
            'summary': None,
            'warning': 'Database not available'
        }

    try:
        summary = RoadmapRatingDB.compute_summary(roadmap_id)
        return {
            'roadmap_id': roadmap_id,
            'summary': summary
        }
    except Exception as e:
        print(f"Summary computation error: {e}")
        return {
            'roadmap_id': roadmap_id,
            'summary': None,
            'error': str(e)
        }


@app.post('/api/lora/train')
def lora_train(payload: dict):
    """
    Start LoRA fine-tuning on career-related Q&A dataset.
    
    This is a long-running task. In production, this should be async
    or queued to a background job service.
    """
    try:
        from backend.llm.lora_dataset import generate_qa_pairs
        from backend.llm.lora_train import LoRATrainer
    except ImportError:
        return {
            'success': False,
            'error': 'LoRA modules not available. Install with: pip install transformers peft torch'
        }
    
    # Get parameters
    num_samples = int(payload.get('num_samples', 50))
    num_epochs = int(payload.get('num_epochs', 3))
    batch_size = int(payload.get('batch_size', 8))
    model_name = payload.get('model_name', 'distilgpt2')
    
    try:
        # Generate dataset
        qa_pairs = generate_qa_pairs(n=num_samples, seed=42)
        
        # Initialize trainer
        trainer = LoRATrainer(model_name=model_name)
        
        # Load model with LoRA
        if not trainer.load_model():
            return {
                'success': False,
                'error': f'Failed to load model: {model_name}'
            }
        
        # Prepare dataset
        train_texts, val_texts = trainer.prepare_dataset(qa_pairs)
        
        # Estimate time
        from backend.llm.lora_train import estimate_training_time
        time_estimate = estimate_training_time(
            num_samples=len(train_texts),
            model_size='small',
            has_gpu=trainer.has_gpu
        )
        
        # For demo: don't actually train (too slow without GPU)
        # In production with GPU, uncomment the train call:
        # save_path = trainer.train(train_texts, val_texts, num_epochs=num_epochs, batch_size=batch_size)
        
        return {
            'success': True,
            'message': 'LoRA training configured successfully',
            'config': {
                'model': model_name,
                'num_samples': num_samples,
                'train_size': len(train_texts),
                'val_size': len(val_texts),
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'gpu_available': trainer.has_gpu,
            },
            'time_estimate': time_estimate,
            'status': 'ready' if trainer.has_gpu else 'gpu_recommended',
            'note': 'Actual training requires GPU. Use: trainer.train(train_texts, val_texts, ...)'
        }
    
    except Exception as e:
        print(f"LoRA training error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.get('/api/lora/status')
def lora_status():
    """Check LoRA training availability and GPU status."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    except ImportError:
        gpu_available = False
        gpu_name = None
    
    try:
        from backend.llm.lora_train import LoRATrainer
        lora_available = True
    except ImportError:
        lora_available = False
    
    return {
        'lora_available': lora_available,
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
        'recommendation': 'LoRA training with GPU can be 10-100x faster' if not gpu_available else 'GPU available for optimal training'
    }


@app.post('/api/ratings/store')
def store_rating(req: RoadmapRatingRequest):
    """Store a grader's rating for a roadmap."""
    if not DB_AVAILABLE:
        return {
            'success': False,
            'error': 'Database not available. Ratings will not persist.',
            'rating': {
                'user_id': req.user_id,
                'roadmap_id': req.roadmap_id,
                'grader_id': req.grader_id
            }
        }
    
    try:
        rating = RoadmapRatingDB.save_rating(
            user_id=req.user_id,
            roadmap_id=req.roadmap_id,
            grader_id=req.grader_id,
            usefulness=req.usefulness,
            clarity=req.clarity,
            factuality=req.factuality,
            actionability=req.actionability,
            overall_rating=req.overall_rating,
            comments=req.comments
        )
        return {
            'success': True,
            'rating_id': rating.id,
            'message': 'Rating stored successfully'
        }
    except Exception as e:
        print(f"Rating storage error: {e}")
        return {
            'success': False,
            'error': str(e)
        }


@app.get('/api/ratings/list')
def list_ratings(roadmap_id: str):
    """Get all ratings for a specific roadmap."""
    if not DB_AVAILABLE:
        return {
            'roadmap_id': roadmap_id,
            'ratings': [],
            'warning': 'Database not available'
        }
    
    try:
        ratings = RoadmapRatingDB.get_ratings(roadmap_id)
        return {
            'roadmap_id': roadmap_id,
            'total_ratings': len(ratings),
            'ratings': [
                {
                    'id': r.id,
                    'grader_id': r.grader_id,
                    'usefulness': r.usefulness,
                    'clarity': r.clarity,
                    'factuality': r.factuality,
                    'actionability': r.actionability,
                    'overall_rating': r.overall_rating,
                    'comments': r.comments,
                    'created_at': r.created_at.isoformat() if r.created_at else None
                }
                for r in ratings
            ]
        }
    except Exception as e:
        print(f"Rating retrieval error: {e}")
        return {
            'roadmap_id': roadmap_id,
            'ratings': [],
            'error': str(e)
        }


@app.get('/api/ratings/summary')
def rating_summary(roadmap_id: str):
    """Compute summary statistics (mean ± stddev) for a roadmap's ratings."""
    if not DB_AVAILABLE:
        return {
            'roadmap_id': roadmap_id,
            'summary': None,
            'warning': 'Database not available'
        }
    
    try:
        summary = RoadmapRatingDB.compute_summary(roadmap_id)
        return {
            'roadmap_id': roadmap_id,
            'summary': summary
        }
    except Exception as e:
        print(f"Summary computation error: {e}")
        return {
            'roadmap_id': roadmap_id,
            'summary': None,
            'error': str(e)
        }
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import sys

# Ensure backend module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import agent modules; if they fail, we'll use demo responses
try:
    from backend.agents.parse_agent import parse_resume
    from backend.agents.match_agent import match_roles
    from backend.agents.retrieval_agent import retrieve
    from backend.agents.gen_agent import generate_roadmap_chain
    from backend.agents.verify_agent import verify_links
    from backend.agents.psychometric_agent import score_responses
    AGENTS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import agents: {e}. Using demo responses only.")
    AGENTS_AVAILABLE = False

# Import database modules
try:
    from backend.schema import init_db
    from backend.db import PsychometricDB, RoadmapRatingDB
    init_db()  # Initialize database on startup
    DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not initialize database: {e}. Ratings will not persist.")
    DB_AVAILABLE = False

DEMO_PATH = os.path.join(os.path.dirname(__file__), 'demo', 'canned_outputs.json')

app = FastAPI(title='CareerPath-Gen Backend')


class AnalyzeRequest(BaseModel):
    resumeText: str
    targetRole: str
    weeklyHours: int
    psychometricProfile: Optional[dict] = None
    ragEnabled: bool = True
    tone: str = 'friendly'
    length: str = 'detailed'
    showDraftFinal: bool = False
    useLora: bool = False


class RoadmapRatingRequest(BaseModel):
    user_id: str
    roadmap_id: str
    grader_id: str
    usefulness: Optional[int] = None
    clarity: Optional[int] = None
    factuality: Optional[int] = None
    actionability: Optional[int] = None
    overall_rating: Optional[int] = None
    comments: Optional[str] = None


class RoleRecRequest(BaseModel):
    resumeText: str
    interests: str
    workStyle: str


@app.get('/api/health')
def health():
    return {
        'ok': True, 
        'status': 'backend running',
        'agentsAvailable': AGENTS_AVAILABLE,
        'mode': 'production' if AGENTS_AVAILABLE else 'demo'
    }


@app.post('/api/parse')
def api_parse(req: AnalyzeRequest):
    """Parse resume (demo mode)."""
    return {
        'summary': req.resumeText[:200],
        'skills': ['Python', 'SQL', 'JavaScript']
    }


@app.post('/api/match')
def api_match(req: AnalyzeRequest):
    """Match roles using real agent or demo."""
    if AGENTS_AVAILABLE:
        try:
            parsed = parse_resume(req.resumeText)
            matches = match_roles(parsed.get('skills', []), req.targetRole)
            return {
                'matches': [
                    {'role': m.get('role', 'Unknown'), 'matchScore': int(m.get('overlap_score', 0)*100), 'reason': f"Matched {m.get('overlap_score', 0):.0%} of skills"}
                    for m in matches
                ]
            }
        except Exception as e:
            print(f"Match agent error: {e}")
    
    # Demo/fallback response
    return {'matches': [
        {'role': 'Data Analyst', 'matchScore': 85, 'reason': 'Good fit'},
        {'role': 'Backend Engineer', 'matchScore': 72, 'reason': 'Some overlap'}
    ]}


@app.post('/api/retrieve')
def api_retrieve(q: dict):
    """Retrieve RAG context using real agent or demo."""
    query = q.get('query', '')
    
    if AGENTS_AVAILABLE:
        try:
            results = retrieve(query, k=6)
            return {'results': results}
        except Exception as e:
            print(f"Retrieve agent error: {e}")
    
    # Demo/fallback response
    return {'results': [
        {'id': 'r1', 'text': 'Python fundamentals and core concepts...', 'score': 0.9},
        {'id': 'r2', 'text': 'Data structures and algorithms...', 'score': 0.85}
    ]}


@app.post('/api/analyze')
def api_analyze(req: AnalyzeRequest):
    """Analyze career path using real agents or demo responses."""
    # Try to use real agents if available
    if AGENTS_AVAILABLE:
        try:
            parsed = parse_resume(req.resumeText)
            matches = match_roles(parsed.get('skills', []), req.targetRole)
            
            # Use RAG if enabled
            retrieved = []
            if req.ragEnabled:
                retrieved = retrieve(f"{req.targetRole} {' '.join(parsed.get('skills', []))}", k=6)
            
            # Generate roadmap with controls
            draft, critique, final = generate_roadmap_chain(
                parsed, 
                req.targetRole, 
                req.weeklyHours, 
                retrieved,
                req.psychometricProfile,
                tone=req.tone,
                length=req.length
            )
            
            # Only return draft if showDraftFinal enabled, else just final
            result = {
                'matchScore': 75,
                'matchSummary': f'Career roadmap for {req.targetRole}',
                'resumeSummary': parsed.get('summary', req.resumeText[:100]),
                'currentSkills': parsed.get('skills', []),
                'missingSkills': [],
                'roadmap': [],
                'immediateActions': ['Start with fundamentals', 'Build projects', 'Network with professionals'],
                'interviewTips': ['Practice system design', 'Review company culture', 'Prepare behavioral stories'],
                'final_roadmap': final,
            }
            if req.showDraftFinal:
                result['draft_roadmap'] = draft
                result['critique'] = critique
            
            return result
        except Exception as e:
            print(f"Agent error: {e}")
            # Fall through to demo mode
    
    # Demo/fallback response
    if os.path.exists(DEMO_PATH):
        try:
            with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                canned = json.load(f)
            return canned.get('analysis', {})
        except Exception:
            pass
    
    return {
        'matchScore': 75,
        'matchSummary': f'Career roadmap for {req.targetRole}',
        'resumeSummary': req.resumeText[:100],
        'currentSkills': ['Python', 'SQL'],
        'missingSkills': [],
        'roadmap': [],
        'immediateActions': ['Learn target skills', 'Build portfolio', 'Network'],
        'interviewTips': ['Prepare technical questions', 'Research company', 'Practice interviews']
    }


@app.post('/api/recommend')
def api_recommend(req: RoleRecRequest):
    """Recommend roles using real agent or demo."""
    if AGENTS_AVAILABLE:
        try:
            parsed = parse_resume(req.resumeText) if req.resumeText else {}
            matches = match_roles(parsed.get('skills', []), '')
            return {
                'recommendations': [
                    {'role': m.get('role', 'Unknown'), 'matchScore': int(m.get('overlap_score', 0)*100), 'reason': f"Matched {m.get('overlap_score', 0):.0%} of skills"}
                    for m in matches[:5]
                ]
            }
        except Exception as e:
            print(f"Recommend agent error: {e}")
    
    # Demo/fallback response
    if os.path.exists(DEMO_PATH):
        try:
            with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                canned = json.load(f)
            return canned.get('recommendations', {'recommendations': []})
        except Exception:
            pass
    
    return {'recommendations': [
        {'role': 'Data Analyst', 'matchScore': 85, 'reason': 'Good skill overlap'},
        {'role': 'Machine Learning Engineer', 'matchScore': 78, 'reason': 'Strong technical background'}
    ]}


@app.post('/api/verify-links')
def api_verify(payload: dict):
    """Verify links using real agent or demo."""
    links = payload.get('links', [])
    
    if AGENTS_AVAILABLE:
        try:
            results = verify_links(links)
            return {'results': results}
        except Exception as e:
            print(f"Verify agent error: {e}")
    
    # Demo/fallback response
    return {'results': [{'url': link, 'ok': True, 'status': 200} for link in links]}


@app.get('/api/psychometric/questions')
def psychometric_questions():
    """Get psychometric questions."""
    return {
        'questions': [
            {'id': 1, 'text': 'When facing a completely new problem, what is your first instinct?', 'options': ['theory','practical','visual','social']},
            {'id': 2, 'text': 'In a team setting, which role do you naturally gravitate towards?', 'options': ['structure','deep_work','connector','driver']},
            {'id': 3, 'text': 'How do you prefer to learn a new technology?', 'options': ['docs','video','project']},
            {'id': 4, 'text': 'What motivates you most in a career?', 'options': ['mastery','creation','logic']},
        ]
    }


@app.post('/api/psychometric/score')
def psychometric_score(payload: dict):
    """Score psychometric responses using real agent or demo, and save to database."""
    responses = payload.get('responses', {})
    user_id = payload.get('user_id', 'anonymous')
    
    profile = None
    
    if AGENTS_AVAILABLE:
        try:
            profile = score_responses(responses)
        except Exception as e:
            print(f"Psychometric agent error: {e}")
    
    # Demo/fallback response
    if not profile:
        if os.path.exists(DEMO_PATH):
            try:
                with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                    canned = json.load(f)
                profile = canned.get('psychometric', {}).get('profile', {})
            except Exception:
                pass
        
        if not profile:
            profile = {
                'archetype': 'Strategic Architect',
                'learningStyle': 'Practical',
                'workStyle': 'Collaborative',
                'traits': ['Analytical', 'Pragmatic'],
                'description': 'Strategic Architect who learns best via Practical methods.'
            }
    
    # Save to database if available
    if DB_AVAILABLE:
        try:
            PsychometricDB.save_score(
                user_id=user_id,
                test_responses=responses,
                archetype=profile.get('archetype', ''),
                learning_style=profile.get('learningStyle', ''),
                work_style=profile.get('workStyle', ''),
                traits=profile.get('traits', []),
                description=profile.get('description', '')
            )
        except Exception as e:
            print(f"Database save error: {e}")
    
    return {'profile': profile}


@app.post('/api/synthetic/resumes')
def gen_synthetic(payload: dict):
    """Generate synthetic resumes using real agent or demo."""
    n = int(payload.get('n', 10))
    
    if AGENTS_AVAILABLE:
        try:
            resumes = generate_synthetic_resumes(n=n)
            return {'generated': resumes}
        except Exception as e:
            print(f"Synthetic agent error: {e}")
    
    # Demo/fallback response
    if os.path.exists(DEMO_PATH):
        try:
            with open(DEMO_PATH, 'r', encoding='utf-8') as f:
                canned = json.load(f)
            return canned.get('synthetic', {'generated': []})
        except Exception:
            pass
    
    return {'generated': [
        {'summary': f'Synthetic resume {i+1}', 'skills': 'Python,SQL', 'projects': ['Project 1', 'Project 2']}
        for i in range(n)
    ]}
