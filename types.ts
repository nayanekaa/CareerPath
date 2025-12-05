export enum PriorityLevel {
  HIGH = "High",
  MEDIUM = "Medium",
  LOW = "Low"
}

export interface MissingSkill {
  skill: string;
  priority: PriorityLevel;
  reason: string;
}

export interface RoadmapWeek {
  week: string;
  theme: string;
  tasks: string[];
  resources: string[];
}

export interface RoadmapPhase {
  phaseTitle: string;
  milestoneName: string; // New: e.g. "Junior Developer Competency"
  estimatedSalary: string; // New: e.g. "$60k - $80k"
  weeks: RoadmapWeek[];
}

export interface PsychometricProfile {
  archetype: string;
  learningStyle: 'Visual' | 'Practical' | 'Theoretical' | 'Social';
  workStyle: 'Independent' | 'Collaborative' | 'Structured' | 'Adaptive';
  traits: string[];
  description: string;
}

export interface CareerAnalysisResult {
  matchScore: number;
  matchSummary: string;
  resumeSummary: string;
  currentSkills: string[];
  missingSkills: MissingSkill[];
  roadmap: RoadmapPhase[];
  immediateActions: string[];
  interviewTips: string[];
  psychometricAnalysis?: string;
}

export interface UserInput {
  resumeText: string;
  targetRole: string;
  weeklyHours: number;
  psychometricProfile?: PsychometricProfile;
}

export interface RoleRecommendation {
  role: string;
  matchScore: number;
  reason: string;
}