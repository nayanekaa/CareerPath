import { GoogleGenAI, Type, Schema } from "@google/genai";
import { CareerAnalysisResult, RoleRecommendation, PsychometricProfile } from "../types";
import * as backendService from './backendService';

const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || '').trim();

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const analysisSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    matchScore: {
      type: Type.NUMBER,
      description: "A score from 0 to 100 indicating how well the candidate fits the role.",
    },
    matchSummary: {
      type: Type.STRING,
      description: "A concise 1-sentence summary of the fit analysis.",
    },
    resumeSummary: {
      type: Type.STRING,
      description: "A brief summary of the candidate's background extracted from the resume.",
    },
    currentSkills: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "List of relevant canonical skills found in the resume.",
    },
    missingSkills: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          skill: { type: Type.STRING },
          priority: { type: Type.STRING, enum: ["High", "Medium", "Low"] },
          reason: { type: Type.STRING, description: "Why this skill is critical for the target role." },
        },
        required: ["skill", "priority", "reason"],
      },
      description: "List of top skills the candidate lacks, ranked by priority.",
    },
    roadmap: {
      type: Type.ARRAY,
      description: "A learning roadmap split into phases, treated as career milestones.",
      items: {
        type: Type.OBJECT,
        properties: {
          phaseTitle: { type: Type.STRING, description: "e.g., 'Phase 1: Foundations'" },
          milestoneName: { type: Type.STRING, description: "A catchy job-title-like milestone name for this phase (e.g., 'Junior Pythonista' or 'Backend Capable')." },
          estimatedSalary: { type: Type.STRING, description: "Estimated market salary range for someone at this specific level of competency (e.g. '$50k-$70k')." },
          weeks: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                week: { type: Type.STRING, description: "e.g., 'Week 1'" },
                theme: { type: Type.STRING, description: "Main topic for the week" },
                tasks: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Actionable bullet points." },
                resources: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Resource links." },
              },
              required: ["week", "theme", "tasks", "resources"],
            },
          },
        },
        required: ["phaseTitle", "weeks", "milestoneName", "estimatedSalary"],
      },
    },
    immediateActions: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "3 very specific immediate actions the user can take today.",
    },
    interviewTips: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: "3 specific interview preparation tips.",
    },
    psychometricAnalysis: {
      type: Type.STRING,
      description: "A short sentence explaining how the roadmap was adapted to their psychometric profile."
    }
  },
  required: [
    "matchScore",
    "matchSummary",
    "resumeSummary",
    "currentSkills",
    "missingSkills",
    "roadmap",
    "immediateActions",
    "interviewTips",
  ],
};

const recommendationSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    recommendations: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          role: { type: Type.STRING, description: "A specific job title." },
          matchScore: { type: Type.NUMBER, description: "Estimated fit score (0-100)." },
          reason: { type: Type.STRING, description: "Why this role is a good fit." },
        },
        required: ["role", "matchScore", "reason"],
      },
    },
  },
  required: ["recommendations"],
};

export const analyzeCareerPath = async (
  resumeText: string,
  targetRole: string,
  weeklyHours: number,
  psychometricProfile?: PsychometricProfile
): Promise<CareerAnalysisResult> => {
  try {
    // If a local backend is configured, prefer it (demo/dev friendly)
    if (BACKEND_URL) {
      return await backendService.analyzeCareerPath({ resumeText, targetRole, weeklyHours, psychometricProfile });
    }
    let psychometricContext = "";
    if (psychometricProfile) {
      psychometricContext = `
        PSYCHOMETRIC PROFILE:
        - Archetype: ${psychometricProfile.archetype}
        - Learning Style: ${psychometricProfile.learningStyle}
        - Work Style: ${psychometricProfile.workStyle}
        - Traits: ${psychometricProfile.traits.join(", ")}
        
        ADAPTATION INSTRUCTION:
        Tailor the roadmap tone and resources. 
        If 'Visual', suggest videos. If 'Practical', suggest projects.
        The "Milestone Names" should reflect their archetype (e.g. "Architect" vs "Builder").
      `;
    }

    const prompt = `
      You are PATHFINDER, a futuristic career architect.
      
      User details:
      - Target role: "${targetRole}"
      - Weekly available hours: ${weeklyHours}
      - RESUME: "${resumeText.slice(0, 15000)}"

      ${psychometricContext}

      Perform a gap analysis.
      
      Output strict JSON matching the schema.
      For the "roadmap", break the journey into 3 distinct Phases (Milestones).
      Each Phase should have a "milestoneName" (e.g. "Entry-Level Capability") and an "estimatedSalary" that reflects the market value of someone at that specific stage of the learning path.
      
      Be inspiring yet realistic.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: analysisSchema,
        temperature: 0.4,
      },
    });

    const text = response.text;
    if (!text) throw new Error("No response from AI");

    return JSON.parse(text) as CareerAnalysisResult;
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    throw new Error("Failed to generate career path. Please try again.");
  }
};

export const getRoleRecommendations = async (
  resumeText: string,
  interests: string,
  workStyle: string
): Promise<RoleRecommendation[]> => {
  try {
    const prompt = `
      You are PATHFINDER. Recommend 3 distinct job roles.
      
      Interests: "${interests}"
      Work Style: "${workStyle}"
      Resume Snippet: "${resumeText.slice(0, 5000)}"

      Output strict JSON.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: recommendationSchema,
        temperature: 0.5,
      },
    });

    const text = response.text;
    if (!text) throw new Error("No response from AI");
    
    const data = JSON.parse(text) as { recommendations: RoleRecommendation[] };
    return data.recommendations;
  } catch (error) {
    console.error("Role Recommendation Error:", error);
    throw new Error("Failed to recommend roles.");
  }
};