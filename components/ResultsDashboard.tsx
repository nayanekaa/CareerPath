import React from 'react';
import { CareerAnalysisResult, PriorityLevel, RoadmapPhase } from '../types';
import { CheckCircle2, AlertTriangle, BookOpen, Lightbulb, TrendingUp, User, Target, Zap, Brain, Rocket, DollarSign, Calendar } from 'lucide-react';
import TechStack from './TechStack';

interface ResultsDashboardProps {
  data: CareerAnalysisResult;
  onReset: () => void;
}

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ data, onReset }) => {
  const [roadmapId] = React.useState(Math.random().toString(36).substring(7));
  const [ratingSubmitted, setRatingSubmitted] = React.useState(false);
  const [loraStatus, setLoraStatus] = React.useState<any>(null);

  const handleRateRoadmap = async (rating: number, comments: string) => {
    try {
      const response = await fetch('http://127.0.0.1:8001/api/ratings/store', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'current-user',
          roadmap_id: roadmapId,
          grader_id: 'grader-' + Math.random().toString(36).substring(7),
          usefulness: rating,
          clarity: rating,
          factuality: rating,
          actionability: rating,
          overall_rating: rating,
          comments
        })
      });
      if (response.ok) {
        setRatingSubmitted(true);
      }
    } catch (error) {
      console.error('Rating submission failed:', error);
    }
  };

  React.useEffect(() => {
    const checkLoraStatus = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8001/api/lora/status');
        if (response.ok) {
          setLoraStatus(await response.json());
        }
      } catch (error) {
        console.error('LoRA status check failed:', error);
      }
    };
    checkLoraStatus();
  }, []);

  return (
    <div className="w-full max-w-6xl mx-auto pb-20 space-y-16 animate-fade-in-up">
      
      {/* HEADER SUMMARY SECTION */}
      <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
        
        {/* SCORE CARD */}
        <div className="md:col-span-4 bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8 flex flex-col items-center justify-center relative overflow-hidden group">
           <div className="absolute inset-0 bg-gradient-to-br from-violet-500/10 to-transparent opacity-50"></div>
           <div className="relative z-10 text-center">
             <div className="inline-flex items-center justify-center w-24 h-24 rounded-full border-4 border-slate-800 bg-slate-900 mb-4 shadow-[0_0_30px_rgba(139,92,246,0.3)] relative">
                <span className={`text-4xl font-extrabold ${data.matchScore > 75 ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {data.matchScore}%
                </span>
                <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="46" stroke="currentColor" strokeWidth="2" fill="none" className="text-slate-800" />
                  <circle cx="50" cy="50" r="46" stroke="currentColor" strokeWidth="2" fill="none" 
                    className={`${data.matchScore > 75 ? 'text-emerald-500' : 'text-amber-500'} transition-all duration-1000 ease-out`}
                    strokeDasharray="289"
                    strokeDashoffset={289 - (289 * data.matchScore) / 100}
                    strokeLinecap="round"
                  />
                </svg>
             </div>
             <h3 className="text-lg font-bold text-white mb-1">Role Fit Probability</h3>
             <p className="text-xs text-slate-400">{data.matchSummary}</p>
           </div>
        </div>

        {/* RESUME & GAPS */}
        <div className="md:col-span-8 grid grid-cols-1 gap-6">
           <div className="bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-6 relative">
              <h4 className="text-xs font-bold text-violet-400 uppercase tracking-widest mb-3 flex items-center">
                <User className="w-4 h-4 mr-2" /> Profile Analysis
              </h4>
              <p className="text-sm text-slate-300 italic mb-4 leading-relaxed">"{data.resumeSummary}"</p>
              <div className="flex flex-wrap gap-2">
                 {data.currentSkills.slice(0, 6).map((skill, idx) => (
                   <span key={idx} className="px-2 py-1 bg-white/5 border border-white/10 rounded text-[10px] text-slate-300">
                     {skill}
                   </span>
                 ))}
              </div>
           </div>

           <div className="bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-6 relative overflow-hidden">
              <div className="absolute right-0 top-0 p-6 opacity-10">
                <AlertTriangle className="w-24 h-24 text-red-500" />
              </div>
              <h4 className="text-xs font-bold text-red-400 uppercase tracking-widest mb-4 flex items-center">
                 <Target className="w-4 h-4 mr-2" /> Critical Gaps
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 relative z-10">
                {data.missingSkills.slice(0, 4).map((gap, idx) => (
                  <div key={idx} className="flex items-center space-x-3 p-3 rounded-xl bg-red-500/5 border border-red-500/10 hover:border-red-500/30 transition-colors">
                     <div className={`w-1.5 h-1.5 rounded-full ${gap.priority === PriorityLevel.HIGH ? 'bg-red-500 shadow-[0_0_8px_red]' : 'bg-amber-500'}`} />
                     <div>
                       <p className="text-sm font-bold text-slate-200">{gap.skill}</p>
                       <p className="text-[10px] text-slate-500">{gap.priority} Priority</p>
                     </div>
                  </div>
                ))}
              </div>
           </div>
        </div>
      </div>

      {/* ROADMAP VISUALIZATION (VERTICAL TIMELINE) */}
      <div className="relative">
        <div className="text-center mb-16 space-y-4">
           <h2 className="text-3xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-cyan-400 inline-block">
             Trajectory Calculated
           </h2>
           <p className="text-slate-400 text-sm max-w-md mx-auto">
             Follow the path below to bridge your skill gaps and reach your target salary band.
           </p>
        </div>

        {/* Central Line */}
        <div className="absolute left-4 md:left-1/2 top-24 bottom-24 w-0.5 bg-gradient-to-b from-violet-500 via-cyan-500 to-slate-800 hidden md:block opacity-30"></div>

        <div className="space-y-12 relative">
          {data.roadmap.map((phase, idx) => (
            <TimelineNode key={idx} phase={phase} index={idx} total={data.roadmap.length} />
          ))}
          
          {/* Target Node */}
          <div className="relative md:flex md:justify-center animate-fade-in-up delay-500">
             <div className="md:w-1/2 p-4 flex flex-col items-center">
                <div className="w-16 h-16 bg-gradient-to-br from-emerald-500 to-cyan-500 rounded-full flex items-center justify-center shadow-[0_0_40px_rgba(16,185,129,0.4)] z-10 mb-4 border-4 border-slate-950">
                  <Rocket className="w-8 h-8 text-white" />
                </div>
                <div className="text-center">
                  <h3 className="text-2xl font-bold text-white">Target Reached</h3>
                  <p className="text-emerald-400 font-medium">{data.roadmap[data.roadmap.length-1].estimatedSalary}+</p>
                </div>
             </div>
          </div>
        </div>
      </div>

      {/* IMMEDIATE ACTIONS GRID */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-gradient-to-br from-violet-900/40 to-slate-900/40 border border-violet-500/20 rounded-3xl p-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-violet-500 blur-[80px] opacity-20 rounded-full pointer-events-none"></div>
          <h3 className="text-lg font-bold text-white mb-6 flex items-center">
             <Zap className="w-5 h-5 text-yellow-400 mr-2" /> Immediate Actions
          </h3>
          <div className="space-y-4">
            {data.immediateActions.map((action, idx) => (
              <div key={idx} className="flex items-start p-4 bg-white/5 rounded-xl border border-white/5 hover:bg-white/10 transition-colors">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-violet-500/20 text-violet-300 flex items-center justify-center text-xs font-bold mr-3 mt-0.5">
                  {idx + 1}
                </span>
                <p className="text-sm text-slate-300 font-medium leading-relaxed">{action}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gradient-to-br from-cyan-900/40 to-slate-900/40 border border-cyan-500/20 rounded-3xl p-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500 blur-[80px] opacity-20 rounded-full pointer-events-none"></div>
          <h3 className="text-lg font-bold text-white mb-6 flex items-center">
             <Lightbulb className="w-5 h-5 text-cyan-400 mr-2" /> Interview Prep
          </h3>
          <div className="space-y-4">
            {data.interviewTips.map((tip, idx) => (
              <div key={idx} className="flex items-start p-4 bg-white/5 rounded-xl border border-white/5 hover:bg-white/10 transition-colors">
                 <div className="mr-3 mt-1">
                   <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-[0_0_5px_cyan]"></div>
                 </div>
                 <p className="text-sm text-slate-300 font-medium leading-relaxed">{tip}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <TechStack />
    </div>
  );
};

const TimelineNode: React.FC<{ phase: RoadmapPhase; index: number; total: number }> = ({ phase, index, total }) => {
  const isLeft = index % 2 === 0;

  return (
    <div className={`relative flex flex-col md:flex-row items-center ${isLeft ? 'md:flex-row' : 'md:flex-row-reverse'} w-full group`}>
      
      {/* Connector Dot */}
      <div className="absolute left-4 md:left-1/2 w-4 h-4 rounded-full bg-slate-950 border-2 border-cyan-500 shadow-[0_0_10px_cyan] z-20 -translate-x-2 md:-translate-x-2 mt-0 md:mt-0 flex items-center justify-center">
         <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
      </div>

      {/* Spacer for alternate side */}
      <div className="hidden md:block w-1/2"></div>

      {/* Content Card */}
      <div className="w-full md:w-1/2 pl-12 md:pl-0 md:p-8 z-10">
         <div className={`
           bg-slate-900/60 backdrop-blur-md border border-white/10 p-6 rounded-2xl shadow-xl transition-all duration-500 hover:border-violet-500/40 hover:shadow-[0_0_30px_rgba(139,92,246,0.15)] hover:-translate-y-1 relative
           ${isLeft ? 'md:mr-8' : 'md:ml-8'}
         `}>
            {/* Phase Badge */}
            <div className="absolute -top-3 left-6 bg-slate-950 px-3 py-1 border border-violet-500/30 rounded-full shadow-lg flex items-center">
               <span className="text-[10px] font-bold text-violet-300 uppercase tracking-wider">{phase.phaseTitle}</span>
            </div>

            <div className="mt-2 mb-4">
               <h3 className="text-xl font-bold text-white mb-1 group-hover:text-cyan-400 transition-colors">{phase.milestoneName}</h3>
               <div className="flex items-center text-emerald-400 text-xs font-mono bg-emerald-900/20 w-fit px-2 py-1 rounded border border-emerald-500/20">
                  <DollarSign className="w-3 h-3 mr-1" /> {phase.estimatedSalary}
              </div>
               {phase.weeks.map((week, idx) => (
              {/* RATING SECTION */}
              <div className="bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8">
                <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                  <Lightbulb className="w-5 h-5 mr-2 text-amber-400" />
                  Rate This Roadmap
                </h3>
                {ratingSubmitted ? (
                  <div className="bg-emerald-900/20 border border-emerald-500/20 rounded-xl p-4 text-emerald-300 text-sm">
                    ✓ Thank you for rating! Your feedback helps improve the AI.
                  </div>
                ) : (
                  <div className="space-y-4">
                    <p className="text-sm text-slate-300">How useful was this career roadmap?</p>
                    <div className="flex gap-2">
                      {[1, 2, 3, 4, 5].map(rating => (
                        <button
                          key={rating}
                          onClick={() => handleRateRoadmap(rating, 'Good roadmap')}
                          className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-violet-500 text-white transition-colors"
                        >
                          {rating}★
                        </button>
                      ))}
                    </div>
                    <textarea
                      placeholder="Optional: Share your feedback..."
                      className="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-sm text-white placeholder-slate-500"
                      rows={2}
                    />
                  </div>
                )}
              </div>
                 <div key={idx} className="border-l-2 border-slate-700 pl-4 py-1 hover:border-cyan-500 transition-colors">
              {/* LORA STATUS SECTION */}
              {loraStatus && (
                <div className="bg-slate-900/40 backdrop-blur-xl border border-white/10 rounded-3xl p-8">
                  <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                    <Brain className="w-5 h-5 mr-2 text-blue-400" />
                    LoRA Model Fine-tuning
                  </h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-slate-800/50 rounded-lg p-4">
                      <p className="text-xs text-slate-400">LoRA Available</p>
                      <p className="text-lg font-bold text-blue-400">{loraStatus.lora_available ? '✓ Yes' : '✗ No'}</p>
                    </div>
                    <div className="bg-slate-800/50 rounded-lg p-4">
                      <p className="text-xs text-slate-400">GPU Available</p>
                      <p className="text-lg font-bold text-green-400">{loraStatus.gpu_available ? '✓ Yes' : '✗ No'}</p>
                    </div>
                  </div>
                  {loraStatus.gpu_name && (
                    <p className="text-xs text-slate-400 mt-3">GPU: {loraStatus.gpu_name}</p>
                  )}
                  <p className="text-xs text-slate-400 mt-3">{loraStatus.recommendation}</p>
                </div>
              )}
            </div>
          );
        };
                    <p className="text-xs font-bold text-slate-400 uppercase mb-1">{week.week}: {week.theme}</p>
                    <ul className="space-y-1">
                      {week.tasks.slice(0, 2).map((task, tIdx) => (
                        <li key={tIdx} className="text-sm text-slate-300 truncate">• {task}</li>
                      ))}
                    </ul>
                 </div>
               ))}
            </div>
         </div>
      </div>
    </div>
  );
};

export default ResultsDashboard;