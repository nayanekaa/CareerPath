import React, { useState } from 'react';
import InputSection from './components/InputSection';
import ResultsDashboard from './components/ResultsDashboard';
import PsychometricTest from './components/PsychometricTest';
import { analyzeCareerPath } from './services/geminiService';
import { CareerAnalysisResult, UserInput, PsychometricProfile } from './types';
import { ArrowRight, Brain, Compass, Stars } from 'lucide-react';

type AppStep = 'intro' | 'psychometric' | 'input' | 'results';

const App: React.FC = () => {
  const [step, setStep] = useState<AppStep>('intro');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CareerAnalysisResult | null>(null);
  const [psychProfile, setPsychProfile] = useState<PsychometricProfile | undefined>(undefined);

  const handlePsychometricComplete = (profile: PsychometricProfile) => {
    setPsychProfile(profile);
    setStep('input');
  };

  const handleSubmit = async (input: UserInput) => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeCareerPath(
        input.resumeText,
        input.targetRole,
        input.weeklyHours,
        psychProfile
      );
      setResult(data);
      setStep('results');
    } catch (err: any) {
      setError(err.message || "Navigation system offline. Please retry.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setStep('intro');
    setPsychProfile(undefined);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-cyan-500/30 font-inter relative overflow-x-hidden">
      {/* Background Ambient Glows */}
      <div className="fixed top-[-10%] left-[-10%] w-[40%] h-[40%] bg-violet-900/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="fixed bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-900/20 rounded-full blur-[120px] pointer-events-none" />

      {/* Glass Header */}
      <header className="fixed w-full top-0 z-50 backdrop-blur-xl bg-slate-950/70 border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
           <div className="flex items-center space-x-3 cursor-pointer group" onClick={handleReset}>
             <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-violet-600 to-cyan-500 rounded-lg blur opacity-75 group-hover:opacity-100 transition-opacity"></div>
                <div className="relative w-10 h-10 bg-slate-900 rounded-lg flex items-center justify-center border border-white/10">
                  <Compass className="w-6 h-6 text-white group-hover:rotate-45 transition-transform duration-500" />
                </div>
             </div>
             <span className="font-bold text-2xl tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
               PATHFINDER
             </span>
           </div>
           
           <div className="flex items-center gap-4">
             {psychProfile && step !== 'intro' && (
                <div className="hidden md:flex items-center px-4 py-1.5 bg-slate-900/50 rounded-full border border-violet-500/30 shadow-[0_0_10px_rgba(139,92,246,0.1)]">
                  <Brain className="w-4 h-4 text-violet-400 mr-2" />
                  <span className="text-xs font-bold text-violet-200 uppercase tracking-wider">{psychProfile.archetype}</span>
                </div>
             )}
           </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 pt-32 pb-20 relative z-10">
        {error && (
          <div className="mb-10 p-4 bg-red-950/50 border border-red-500/50 text-red-200 rounded-2xl flex items-center justify-between animate-fade-in backdrop-blur-md">
             <span className="flex items-center font-medium">
               <svg className="w-5 h-5 mr-3" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
               {error}
             </span>
             <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 transition-colors">&times;</button>
          </div>
        )}

        {/* STEP 1: HERO INTRO */}
        {step === 'intro' && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center animate-fade-in space-y-10">
            <div className="relative">
               <div className="absolute inset-0 bg-cyan-500 blur-[80px] opacity-20 rounded-full"></div>
               <div className="relative p-6 bg-slate-900/50 rounded-3xl border border-white/10 backdrop-blur-md shadow-2xl">
                 <Stars className="w-12 h-12 text-cyan-400" />
               </div>
            </div>
            
            <div className="space-y-6 max-w-4xl">
              <h1 className="text-6xl md:text-7xl font-extrabold tracking-tight text-white leading-tight">
                Where do you <br/>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 via-cyan-400 to-violet-400 animate-gradient-x">
                  want to go?
                </span>
              </h1>
              <p className="text-xl md:text-2xl text-slate-400 max-w-2xl mx-auto font-light leading-relaxed">
                Initialize your career trajectory. <strong className="text-slate-200 font-medium">Pathfinder</strong> analyzes your DNA and builds a high-fidelity roadmap to your target role.
              </p>
            </div>

            <button 
              onClick={() => setStep('psychometric')}
              className="group relative px-10 py-5 bg-white text-slate-950 font-bold text-lg rounded-full shadow-[0_0_40px_rgba(255,255,255,0.3)] hover:shadow-[0_0_60px_rgba(255,255,255,0.5)] transition-all duration-300 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-violet-200 to-cyan-200 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <span className="relative flex items-center z-10">
                Initiate Sequence
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </span>
            </button>
          </div>
        )}

        {/* STEP 2: PSYCHOMETRIC TEST */}
        {step === 'psychometric' && (
          <div className="flex justify-center animate-fade-in-up">
            <PsychometricTest onComplete={handlePsychometricComplete} />
          </div>
        )}

        {/* STEP 3: INPUT RESUME/ROLE (HERO FORM) */}
        {step === 'input' && (
          <div className="flex flex-col items-center justify-center animate-fade-in">
             <div className="mb-10 px-6 py-3 bg-slate-900/60 border border-violet-500/30 rounded-full flex items-center gap-3 backdrop-blur-md shadow-[0_0_20px_rgba(139,92,246,0.15)]">
                <Brain className="w-4 h-4 text-violet-400" />
                <span className="text-sm font-medium text-slate-300">
                  <span className="text-violet-400 font-bold">{psychProfile?.archetype}</span> Protocol Active
                </span>
             </div>
             <InputSection onSubmit={handleSubmit} isLoading={loading} />
          </div>
        )}

        {/* STEP 4: RESULTS DASHBOARD */}
        {step === 'results' && result && (
          <ResultsDashboard data={result} onReset={handleReset} />
        )}
      </main>

      <footer className="w-full border-t border-white/5 py-8 mt-auto backdrop-blur-sm">
         <div className="max-w-7xl mx-auto px-6 flex justify-between items-center text-slate-600 text-xs tracking-widest uppercase">
           <p>System v2.4.0 // Online</p>
           <p>Powered by Gemini 2.5 Flash</p>
         </div>
      </footer>
    </div>
  );
};

export default App;