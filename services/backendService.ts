import { CareerAnalysisResult, RoleRecommendation, PsychometricProfile } from '../types';

const BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

async function _post(path: string, body: any) {
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Backend error ${res.status}`);
  return res.json();
}

export async function analyzeCareerPath(req: { resumeText: string; targetRole: string; weeklyHours: number; psychometricProfile?: PsychometricProfile }): Promise<CareerAnalysisResult> {
  const data = await _post('/api/analyze', req);
  // backend returns richer structure; attempt to map to frontend type if possible
  if (data.final && typeof data.final === 'string') {
    // demo fallback: parse minimal fields
    return {
      matchScore: data.get?.matchScore ?? 50,
      matchSummary: data.get?.matchSummary ?? 'Demo result',
      resumeSummary: req.resumeText.slice(0, 200),
      currentSkills: data.parsed?.skills || [],
      missingSkills: [],
      roadmap: [],
      immediateActions: [],
      interviewTips: [],
    } as any;
  }
  return data as any;
}

export async function getRoleRecommendations({ resumeText, interests, workStyle }: { resumeText: string; interests: string; workStyle: string; }): Promise<RoleRecommendation[]> {
  const data = await _post('/api/recommend', { resumeText, interests, workStyle });
  return data.recommendations || data;
}

export async function getPsychometricQuestions() {
  const res = await fetch(BASE + '/api/psychometric/questions');
  if (!res.ok) throw new Error('Failed to load questions');
  return res.json();
}

export async function generateSynthetic(seed: string, n = 10) {
  const data = await _post('/api/synthetic/resumes', { seed, n });
  return data.generated;
}
