import React from 'react';

const TopHeader = ({ title, userQuery }) => {
  return (
    <header className="w-full h-24 sticky top-0 z-40 flex items-center justify-between px-10 bg-[#faf8ff]/60 dark:bg-slate-950/60 backdrop-blur-3xl border-b border-[#f3f3fd] dark:border-slate-800/50 font-headline">
      <div className="flex items-center gap-6">
        <div className="flex flex-col">
          <h1 className="text-sm font-black text-primary uppercase tracking-[0.2em] mb-0.5">{title}</h1>
          <div className="flex items-center gap-2">
             <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
             <p className="text-[10px] font-bold opacity-40 uppercase tracking-widest">Neural Link Active</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-8">
        {/* Optional Global Status Indicators */}
        <div className="hidden lg:flex items-center gap-6">
           <div className="flex flex-col items-end">
              <p className="text-[10px] font-black uppercase opacity-30 tracking-widest">Memory Sync</p>
              <p className="text-xs font-bold">94.2 GB Available</p>
           </div>
           <div className="w-px h-8 bg-slate-200 dark:bg-slate-800"></div>
           <div className="flex flex-col items-end">
              <p className="text-[10px] font-black uppercase opacity-30 tracking-widest">Latency</p>
              <p className="text-xs font-bold text-primary">12ms</p>
           </div>
        </div>

        <div className="flex items-center gap-4">
          <button className="w-12 h-12 flex items-center justify-center text-slate-400 hover:text-primary transition-colors">
            <span className="material-symbols-outlined text-2xl">search</span>
          </button>
          <button className="w-12 h-12 flex items-center justify-center text-slate-400 hover:text-primary transition-colors">
            <span className="material-symbols-outlined text-2xl">notifications</span>
          </button>
          
          <div className="w-12 h-12 rounded-2xl bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm flex items-center justify-center overflow-hidden group hover:border-primary/50 transition-all cursor-pointer">
              <span className="material-symbols-outlined text-primary group-hover:scale-110 transition-transform">person</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default TopHeader;
