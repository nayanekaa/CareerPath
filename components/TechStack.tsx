import React from 'react';
import { Cpu, Database, Layers, Zap, Code, FileText } from 'lucide-react';

const TechStack: React.FC = () => {
  return (
    <div className="mt-20 border-t border-white/5 pt-10">
      <div className="text-center mb-10">
        <h3 className="text-sm font-bold text-slate-500 uppercase tracking-[0.2em] flex items-center justify-center gap-2">
           System Architecture
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 max-w-5xl mx-auto">
        
        {/* Step 1 */}
        <div className="bg-slate-900/40 p-5 rounded-xl border border-white/5 relative overflow-hidden group hover:border-violet-500/30 transition-all">
          <div className="absolute top-0 right-0 p-3 opacity-5 group-hover:opacity-10 transition-opacity">
            <FileText className="w-16 h-16 text-violet-500" />
          </div>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-violet-400 text-[10px] font-bold uppercase tracking-wider">Ingestion</span>
          </div>
          <h4 className="font-bold text-slate-200 mb-1">Resume Parsing</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Raw text extraction converts unstructured resume data into analysis-ready tokens.
          </p>
        </div>

        {/* Step 2 */}
        <div className="bg-slate-900/40 p-5 rounded-xl border border-white/5 relative overflow-hidden group hover:border-cyan-500/30 transition-all">
           <div className="absolute top-0 right-0 p-3 opacity-5 group-hover:opacity-10 transition-opacity">
            <Layers className="w-16 h-16 text-cyan-500" />
          </div>
           <div className="flex items-center gap-2 mb-2">
            <span className="text-cyan-400 text-[10px] font-bold uppercase tracking-wider">Context</span>
          </div>
          <h4 className="font-bold text-slate-200 mb-1">RAG Context</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Constructing a context window with personality archetypes and job market vectors.
          </p>
        </div>

        {/* Step 3 */}
        <div className="bg-slate-900/40 p-5 rounded-xl border border-white/5 relative overflow-hidden group hover:border-violet-500/30 transition-all">
           <div className="absolute top-0 right-0 p-3 opacity-5 group-hover:opacity-10 transition-opacity">
            <Cpu className="w-16 h-16 text-violet-500" />
          </div>
           <div className="flex items-center gap-2 mb-2">
            <span className="text-violet-400 text-[10px] font-bold uppercase tracking-wider">Inference</span>
          </div>
          <h4 className="font-bold text-slate-200 mb-1">Gemini 2.5 Flash</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Semantic gap analysis and roadmap generation using advanced zero-shot reasoning.
          </p>
        </div>

         {/* Step 4 */}
         <div className="bg-slate-900/40 p-5 rounded-xl border border-white/5 relative overflow-hidden group hover:border-cyan-500/30 transition-all">
           <div className="absolute top-0 right-0 p-3 opacity-5 group-hover:opacity-10 transition-opacity">
            <Code className="w-16 h-16 text-cyan-500" />
          </div>
           <div className="flex items-center gap-2 mb-2">
            <span className="text-cyan-400 text-[10px] font-bold uppercase tracking-wider">Render</span>
          </div>
          <h4 className="font-bold text-slate-200 mb-1">React Viz</h4>
          <p className="text-xs text-slate-500 leading-relaxed">
            Rendering the roadmap graph using strict JSON schema validation.
          </p>
        </div>

      </div>
    </div>
  );
};

export default TechStack;