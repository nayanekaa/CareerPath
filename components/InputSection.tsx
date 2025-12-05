import React, { useState, useRef } from 'react';
import { UserInput, RoleRecommendation } from '../types';
import { getRoleRecommendations } from '../services/geminiService';
import { Upload, Search, ArrowRight, Clock, Sparkles, AlertCircle, FileText, ChevronRight } from 'lucide-react';

interface InputSectionProps {
  onSubmit: (data: UserInput) => void;
  isLoading: boolean;
}

const MIN_RESUME_CHARS = 50;

const SAMPLE_RESUME = `EDUCATION
University of Technology — B.S. Computer Science (2020 - 2024)
- GPA: 3.7/4.0
- Relevant Coursework: Data Structures, Algorithms, Web Development

EXPERIENCE
Junior Web Developer Intern | TechStart Inc. (Summer 2023)
- Built responsive landing pages using HTML, CSS, and vanilla JavaScript.
- Assisted in debugging backend API endpoints in Node.js.
- Collaborated with UI/UX designers to implement pixel-perfect designs.

PROJECTS
Personal Portfolio Site
- Deployed a React-based portfolio on Vercel.
- Implemented dark mode using CSS variables.

Task Manager App
- Built a simple To-Do app using Python (Flask) and SQLite.
- Implemented basic CRUD operations.

SKILLS
- Languages: Python (Basic), JavaScript (Intermediate), HTML/CSS
- Tools: Git, VS Code
`;

const InputSection: React.FC<InputSectionProps> = ({ onSubmit, isLoading }) => {
  const [role, setRole] = useState('');
  const [resume, setResume] = useState('');
  const [hours, setHours] = useState(15);
  const [activeTab, setActiveTab] = useState<'write' | 'upload'>('write');
  const [errors, setErrors] = useState<{ role?: string; resume?: string }>({});
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Survey State
  const [isSurveyOpen, setIsSurveyOpen] = useState(false);
  const [surveyInterests, setSurveyInterests] = useState('');
  const [surveyStyle, setSurveyStyle] = useState('Collaborative');
  const [recommendations, setRecommendations] = useState<RoleRecommendation[]>([]);
  const [isSurveyLoading, setIsSurveyLoading] = useState(false);

  // Generative AI Controls
  const [ragEnabled, setRagEnabled] = useState(true);
  const [tone, setTone] = useState<'friendly' | 'formal' | 'motivational'>('friendly');
  const [length, setLength] = useState<'short' | 'detailed'>('detailed');
  const [showDraftFinal, setShowDraftFinal] = useState(false);
  const [useLora, setUseLora] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        setResume(text);
        if (text.length >= MIN_RESUME_CHARS) {
           setErrors(prev => ({ ...prev, resume: undefined }));
        }
        setActiveTab('write');
      };
      reader.readAsText(file);
    }
  };

  const validateInputs = (): boolean => {
    const newErrors: { role?: string; resume?: string } = {};
    let isValid = true;

    if (!role.trim()) {
      newErrors.role = "Target coordinates required.";
      isValid = false;
    }

    if (!resume.trim()) {
      newErrors.resume = "Source data required.";
      isValid = false;
    } else if (resume.trim().length < MIN_RESUME_CHARS) {
      newErrors.resume = `Insufficient data. Need ${MIN_RESUME_CHARS} chars.`;
      isValid = false;
    }

    setErrors(newErrors);
    return isValid;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateInputs()) {
      onSubmit({ 
        targetRole: role, 
        resumeText: resume, 
        weeklyHours: hours,
        // Pass generative controls
        ragEnabled,
        tone,
        length,
        showDraftFinal,
        useLora
      } as any);
    }
  };

  const handleSurveySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!surveyInterests) return;
    
    setIsSurveyLoading(true);
    try {
      const recs = await getRoleRecommendations(resume, surveyInterests, surveyStyle);
      setRecommendations(recs);
    } catch (error) {
      console.error(error);
    } finally {
      setIsSurveyLoading(false);
    }
  };

  const selectRecommendation = (recRole: string) => {
    setRole(recRole);
    setErrors(prev => ({ ...prev, role: undefined }));
    setIsSurveyOpen(false);
    setRecommendations([]);
  };

  const loadSample = () => {
    setRole("Full Stack Engineer");
    setResume(SAMPLE_RESUME);
    setHours(20);
    setErrors({});
  };

  return (
    <div className="w-full max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
      
      {/* LEFT COLUMN: INPUTS */}
      <div className="lg:col-span-7 space-y-6">
        
        {/* Target Role Card */}
        <div className="bg-slate-900/40 border border-white/10 rounded-3xl p-8 backdrop-blur-xl shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-violet-500 to-cyan-500 opacity-50 group-hover:opacity-100 transition-opacity"></div>
          
          <label className="block text-xs font-bold text-cyan-400 uppercase tracking-widest mb-4 flex justify-between">
            <span>Destination Coordinates</span>
            {!isSurveyOpen && (
              <button onClick={() => setIsSurveyOpen(true)} className="flex items-center text-white/50 hover:text-white transition-colors">
                <Sparkles className="w-3 h-3 mr-2" />
                Unsure?
              </button>
            )}
          </label>

          {!isSurveyOpen ? (
            <div className="relative">
              <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${errors.role ? 'text-red-400' : 'text-slate-500 group-focus-within:text-cyan-400'} transition-colors`} />
              <input
                type="text"
                className={`w-full bg-slate-950/50 border ${errors.role ? 'border-red-500/50' : 'border-white/10 focus:border-cyan-500/50'} rounded-xl py-4 pl-12 pr-4 text-lg text-white placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition-all shadow-inner`}
                placeholder="e.g. Senior Frontend Architect"
                value={role}
                onChange={(e) => {
                  setRole(e.target.value);
                  if (e.target.value) setErrors(prev => ({ ...prev, role: undefined }));
                }}
              />
              {errors.role && <span className="absolute -bottom-6 left-0 text-xs text-red-400 flex items-center"><AlertCircle className="w-3 h-3 mr-1"/>{errors.role}</span>}
            </div>
          ) : (
            /* Survey UI */
            <div className="bg-slate-950/80 rounded-xl p-4 border border-violet-500/30 animate-fade-in">
              <div className="flex justify-between items-center mb-4">
                 <h4 className="text-sm font-bold text-violet-300">Role Discovery</h4>
                 <button onClick={() => setIsSurveyOpen(false)} className="text-xs text-slate-500 hover:text-white">Close</button>
              </div>
              {recommendations.length === 0 ? (
                <form onSubmit={handleSurveySubmit} className="space-y-4">
                  <input
                    type="text"
                    className="w-full bg-slate-900 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-violet-500 outline-none"
                    placeholder="Keywords: e.g. AI, Creative, Logic"
                    value={surveyInterests}
                    onChange={(e) => setSurveyInterests(e.target.value)}
                  />
                  <select 
                    className="w-full bg-slate-900 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:border-violet-500 outline-none"
                    value={surveyStyle}
                    onChange={(e) => setSurveyStyle(e.target.value)}
                  >
                    <option>Collaborative Teamwork</option>
                    <option>Independent Deep Work</option>
                    <option>Leadership & Management</option>
                  </select>
                  <button 
                    type="submit"
                    disabled={isSurveyLoading || !surveyInterests}
                    className="w-full py-2 bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold uppercase tracking-wider rounded-lg transition-colors disabled:opacity-50"
                  >
                    {isSurveyLoading ? 'Scanning...' : 'Find Matches'}
                  </button>
                </form>
              ) : (
                <div className="space-y-2 max-h-48 overflow-y-auto custom-scrollbar">
                  {recommendations.map((rec, idx) => (
                    <div 
                      key={idx}
                      onClick={() => selectRecommendation(rec.role)}
                      className="p-3 rounded-lg bg-slate-900 hover:bg-violet-900/20 border border-white/5 hover:border-violet-500/50 cursor-pointer transition-all group"
                    >
                       <div className="flex justify-between mb-1">
                         <span className="text-sm font-bold text-slate-200 group-hover:text-white">{rec.role}</span>
                         <span className="text-xs font-bold text-emerald-400">{rec.matchScore}%</span>
                       </div>
                       <p className="text-[10px] text-slate-500 group-hover:text-slate-400">{rec.reason}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Resume Card */}
        <div className="bg-slate-900/40 border border-white/10 rounded-3xl p-8 backdrop-blur-xl shadow-2xl relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 to-violet-500 opacity-50 group-hover:opacity-100 transition-opacity"></div>
          
          <div className="flex justify-between items-center mb-4">
            <label className="text-xs font-bold text-violet-400 uppercase tracking-widest">
              Source Data (Resume)
            </label>
            <div className="flex space-x-2">
              <button onClick={() => setActiveTab('write')} className={`px-3 py-1 text-[10px] font-bold uppercase rounded-md transition-colors ${activeTab === 'write' ? 'bg-violet-500/20 text-violet-300' : 'text-slate-600 hover:text-slate-400'}`}>Text</button>
              <button onClick={() => setActiveTab('upload')} className={`px-3 py-1 text-[10px] font-bold uppercase rounded-md transition-colors ${activeTab === 'upload' ? 'bg-violet-500/20 text-violet-300' : 'text-slate-600 hover:text-slate-400'}`}>File</button>
            </div>
          </div>

          {activeTab === 'write' ? (
            <div className="relative">
              <textarea
                className={`w-full h-40 bg-slate-950/50 border ${errors.resume ? 'border-red-500/50' : 'border-white/10 focus:border-violet-500/50'} rounded-xl p-4 text-sm text-slate-300 placeholder-slate-700 focus:outline-none focus:ring-1 focus:ring-violet-500/50 transition-all font-mono resize-none`}
                placeholder="Paste resume content here..."
                value={resume}
                onChange={(e) => {
                   setResume(e.target.value);
                   if (e.target.value.length >= MIN_RESUME_CHARS) setErrors(prev => ({ ...prev, resume: undefined }));
                }}
              />
              <div className="absolute bottom-3 right-3 text-[10px] font-mono text-slate-600">
                {resume.length} chars
              </div>
            </div>
          ) : (
            <div 
              className={`h-40 border-2 border-dashed ${errors.resume ? 'border-red-500/30' : 'border-slate-800 hover:border-violet-500/50'} rounded-xl flex flex-col items-center justify-center cursor-pointer transition-colors bg-slate-950/30`}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="w-8 h-8 text-slate-600 mb-2" />
              <p className="text-xs text-slate-500 font-medium">Upload .txt or .md</p>
              <input type="file" ref={fileInputRef} className="hidden" accept=".txt,.md" onChange={handleFileChange} />
            </div>
          )}
          
          <div className="mt-4 flex justify-between items-center">
            <button onClick={loadSample} className="text-xs text-slate-600 hover:text-cyan-400 transition-colors flex items-center">
              <Sparkles className="w-3 h-3 mr-1" /> Use Sample Data
            </button>
            {errors.resume && <span className="text-xs text-red-400">{errors.resume}</span>}
          </div>
        </div>

      </div>

      {/* RIGHT COLUMN: ACTION & SETTINGS */}
      <div className="lg:col-span-5 space-y-6">
        
        {/* Hours & Stats */}
        <div className="bg-slate-900/40 border border-white/10 rounded-3xl p-8 backdrop-blur-xl shadow-2xl relative">
           <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-6">
              Velocity (Hours/Week)
           </label>
           <div className="flex items-center justify-between mb-2">
             <Clock className="w-5 h-5 text-slate-400" />
             <span className="text-4xl font-light text-white">{hours}</span>
           </div>
           <input
             type="range"
             min="5"
             max="60"
             step="5"
             className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
             value={hours}
             onChange={(e) => setHours(Number(e.target.value))}
           />
           <div className="flex justify-between mt-2 text-[10px] text-slate-600 font-mono">
             <span>CASUAL</span>
             <span>INTENSE</span>
           </div>
        </div>

        {/* Generative AI Controls Panel */}
        <div className="bg-slate-900/40 border border-white/10 rounded-3xl p-6 backdrop-blur-xl shadow-2xl">
          <h3 className="text-xs font-bold text-violet-400 uppercase tracking-widest mb-4">Generative AI Controls</h3>
          <div className="space-y-4">
            {/* RAG Toggle */}
            <div className="flex items-center justify-between p-3 bg-slate-950/30 rounded-lg border border-white/5">
              <label className="text-sm text-slate-300">RAG (Retrieval-Augmented)</label>
              <button
                onClick={() => setRagEnabled(!ragEnabled)}
                className={`w-12 h-6 rounded-full transition-colors ${ragEnabled ? 'bg-cyan-500' : 'bg-slate-700'}`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${ragEnabled ? 'translate-x-6' : 'translate-x-0.5'}`} />
              </button>
            </div>

            {/* Tone Selection */}
            <div className="p-3 bg-slate-950/30 rounded-lg border border-white/5">
              <label className="text-sm text-slate-300 block mb-2">Tone</label>
              <div className="grid grid-cols-3 gap-2">
                {(['friendly', 'formal', 'motivational'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => setTone(t)}
                    className={`py-2 px-2 text-xs font-bold rounded transition-colors ${tone === t ? 'bg-violet-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
                  >
                    {t.charAt(0).toUpperCase() + t.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Length Selection */}
            <div className="p-3 bg-slate-950/30 rounded-lg border border-white/5">
              <label className="text-sm text-slate-300 block mb-2">Length</label>
              <div className="grid grid-cols-2 gap-2">
                {(['short', 'detailed'] as const).map(l => (
                  <button
                    key={l}
                    onClick={() => setLength(l)}
                    className={`py-2 px-2 text-xs font-bold rounded transition-colors ${length === l ? 'bg-cyan-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
                  >
                    {l.charAt(0).toUpperCase() + l.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Draft vs Final Toggle */}
            <div className="flex items-center justify-between p-3 bg-slate-950/30 rounded-lg border border-white/5">
              <label className="text-sm text-slate-300">Show Draft & Final</label>
              <button
                onClick={() => setShowDraftFinal(!showDraftFinal)}
                className={`w-12 h-6 rounded-full transition-colors ${showDraftFinal ? 'bg-emerald-500' : 'bg-slate-700'}`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${showDraftFinal ? 'translate-x-6' : 'translate-x-0.5'}`} />
              </button>
            </div>

            {/* LoRA Fine-tune Toggle */}
            <div className="flex items-center justify-between p-3 bg-slate-950/30 rounded-lg border border-white/5">
              <label className="text-sm text-slate-300">Use Fine-tuned Model</label>
              <button
                onClick={() => setUseLora(!useLora)}
                className={`w-12 h-6 rounded-full transition-colors ${useLora ? 'bg-orange-500' : 'bg-slate-700'}`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${useLora ? 'translate-x-6' : 'translate-x-0.5'}`} />
              </button>
            </div>
          </div>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleSubmit}
          disabled={isLoading || isSurveyOpen}
          className={`w-full py-8 rounded-3xl font-bold text-xl tracking-tight shadow-2xl flex items-center justify-center space-x-3 transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] relative overflow-hidden group ${
             isLoading || isSurveyOpen ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-gradient-to-r from-violet-600 to-cyan-500 text-white shadow-[0_0_30px_rgba(139,92,246,0.4)] hover:shadow-[0_0_50px_rgba(6,182,212,0.6)]'
          }`}
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Calculating Path...</span>
            </div>
          ) : (
            <>
              <span className="relative z-10">Generate Path</span>
              <ChevronRight className="w-6 h-6 relative z-10 group-hover:translate-x-1 transition-transform" />
              {/* Shine Effect */}
              <div className="absolute top-0 -left-full w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12 group-hover:animate-shine" />
            </>
          )}
        </button>

        <div className="p-4 rounded-2xl bg-slate-900/30 border border-white/5 text-center">
          <p className="text-[10px] text-slate-500 uppercase tracking-widest">
            Estimated Processing Time: 4.2s
          </p>
        </div>

      </div>
    </div>
  );
};

export default InputSection;