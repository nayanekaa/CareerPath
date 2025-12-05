import React, { useState } from 'react';
import { PsychometricProfile } from '../types';
import { Brain, ArrowRight, Check } from 'lucide-react';

interface Question {
  id: number;
  text: string;
  options: {
    label: string;
    value: string;
    trait: string;
    styleImpact: Partial<PsychometricProfile>;
  }[];
}

const QUESTIONS: Question[] = [
  {
    id: 1,
    text: "When facing a completely new problem, what is your first instinct?",
    options: [
      { label: "Research the theory and best practices first.", value: "theory", trait: "Analytical", styleImpact: { learningStyle: 'Theoretical' } },
      { label: "Start building/prototyping immediately and fail fast.", value: "practical", trait: "Pragmatic", styleImpact: { learningStyle: 'Practical' } },
      { label: "Look for a diagram or video to visualize the concept.", value: "visual", trait: "Visualizer", styleImpact: { learningStyle: 'Visual' } },
      { label: "Ask a mentor or peer for their experience.", value: "social", trait: "Collaborative", styleImpact: { learningStyle: 'Social' } },
    ]
  },
  {
    id: 2,
    text: "In a team setting, which role do you naturally gravitate towards?",
    options: [
      { label: "The Planner: Organizing timelines and structure.", value: "structure", trait: "Organized", styleImpact: { workStyle: 'Structured' } },
      { label: "The Solver: Deep diving into the hardest technical bug.", value: "deep_work", trait: "Focused", styleImpact: { workStyle: 'Independent' } },
      { label: "The Glue: Connecting people and ideas.", value: "connector", trait: "Empathetic", styleImpact: { workStyle: 'Collaborative' } },
      { label: "The Driver: Pushing for results and shipping.", value: "driver", trait: "Results-Oriented", styleImpact: { workStyle: 'Adaptive' } },
    ]
  },
  {
    id: 3,
    text: "How do you prefer to learn a new technology?",
    options: [
      { label: "Reading documentation and whitepapers.", value: "docs", trait: "Detail-Oriented", styleImpact: { learningStyle: 'Theoretical' } },
      { label: "Watching a crash course or tutorial video.", value: "video", trait: "Visual", styleImpact: { learningStyle: 'Visual' } },
      { label: "Building a side project from scratch.", value: "project", trait: "Builder", styleImpact: { learningStyle: 'Practical' } },
    ]
  },
  {
    id: 4,
    text: "What motivates you most in a career?",
    options: [
      { label: "Mastering a complex domain perfectly.", value: "mastery", trait: "Perfectionist", styleImpact: { } },
      { label: "Creating tangible value and products.", value: "creation", trait: "Creative", styleImpact: { } },
      { label: "Solving clear, structured problems.", value: "logic", trait: "Logical", styleImpact: { } },
    ]
  }
];

interface Props {
  onComplete: (profile: PsychometricProfile) => void;
}

const PsychometricTest: React.FC<Props> = ({ onComplete }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [answers, setAnswers] = useState<any[]>([]);

  const handleAnswer = (option: any) => {
    const newAnswers = [...answers, option];
    setAnswers(newAnswers);

    if (currentStep < QUESTIONS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      calculateProfile(newAnswers);
    }
  };

  const calculateProfile = (finalAnswers: any[]) => {
    const traits = finalAnswers.map(a => a.trait);
    const learningStyle = finalAnswers.find(a => a.styleImpact.learningStyle)?.styleImpact.learningStyle || 'Practical';
    const workStyle = finalAnswers.find(a => a.styleImpact.workStyle)?.styleImpact.workStyle || 'Independent';

    let archetype = "Versatile Generalist";
    if (traits.includes("Analytical") && traits.includes("Detail-Oriented")) archetype = "Strategic Architect";
    if (traits.includes("Pragmatic") && traits.includes("Builder")) archetype = "Pragmatic Builder";
    if (traits.includes("Visualizer") && traits.includes("Creative")) archetype = "Creative Visionary";
    if (traits.includes("Collaborative") && traits.includes("Empathetic")) archetype = "Team Catalyst";

    const profile: PsychometricProfile = {
      archetype,
      learningStyle,
      workStyle,
      traits,
      description: `You are ${archetype}. You learn best via ${learningStyle} methods.`
    };

    onComplete(profile);
  };

  const currentQ = QUESTIONS[currentStep];

  return (
    <div className="w-full max-w-2xl mx-auto bg-slate-900/60 backdrop-blur-xl border border-white/10 rounded-3xl shadow-2xl overflow-hidden animate-fade-in-up">
      {/* Progress Bar */}
      <div className="bg-slate-950/50 p-6 flex justify-between items-center border-b border-white/5">
        <div>
           <h2 className="text-xl font-bold text-white flex items-center gap-2">
             <Brain className="w-5 h-5 text-violet-400" />
             Neural Calibration
           </h2>
           <p className="text-slate-400 text-xs mt-1 font-mono tracking-widest uppercase">Sequence {currentStep + 1} / {QUESTIONS.length}</p>
        </div>
        <div className="flex gap-1">
          {QUESTIONS.map((_, idx) => (
             <div key={idx} className={`h-1 w-8 rounded-full transition-colors ${idx <= currentStep ? 'bg-cyan-500' : 'bg-slate-800'}`} />
          ))}
        </div>
      </div>

      <div className="p-10">
        <h3 className="text-2xl font-light text-white mb-8 leading-relaxed">
          {currentQ.text}
        </h3>

        <div className="space-y-4">
          {currentQ.options.map((opt, idx) => (
            <button
              key={idx}
              onClick={() => handleAnswer(opt)}
              className="w-full text-left p-5 rounded-xl border border-white/5 bg-slate-800/30 hover:bg-violet-900/20 hover:border-violet-500/50 transition-all group flex items-center justify-between"
            >
              <span className="text-slate-300 font-medium group-hover:text-white transition-colors">{opt.label}</span>
              <div className="w-6 h-6 rounded-full border border-slate-600 group-hover:border-violet-400 flex items-center justify-center opacity-50 group-hover:opacity-100">
                 <div className="w-3 h-3 bg-violet-400 rounded-full scale-0 group-hover:scale-100 transition-transform" />
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PsychometricTest;