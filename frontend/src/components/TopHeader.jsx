import React from 'react';

const TopHeader = ({ title, userQuery, onMenuOpen }) => {
  return (
    <header className="w-full h-16 md:h-20 sticky top-0 z-40 flex items-center justify-between px-4 md:px-8 bg-[#faf8ff]/80 dark:bg-slate-950/80 backdrop-blur-3xl border-b border-[#f3f3fd] dark:border-slate-800/50 font-headline">
      <div className="flex items-center gap-3 md:gap-6">
        {/* Hamburger — mobile only */}
        <button
          onClick={onMenuOpen}
          className="md:hidden w-10 h-10 flex items-center justify-center rounded-2xl text-slate-500 hover:text-primary hover:bg-primary/5 transition-all"
          aria-label="Open navigation menu"
        >
          <span className="material-symbols-outlined text-2xl">menu</span>
        </button>

        <div className="flex flex-col">
          <h1 className="text-xs md:text-sm font-black text-primary uppercase tracking-[0.15em] md:tracking-[0.2em] mb-0.5 capitalize">{title}</h1>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <p className="text-[9px] md:text-[10px] font-bold opacity-40 uppercase tracking-widest">Neural Link Active</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 md:gap-8">
        {/* Status indicators — large screens only */}
        <div className="hidden lg:flex items-center gap-6">
          <div className="flex flex-col items-end">
            <p className="text-[10px] font-black uppercase opacity-30 tracking-widest">Memory Sync</p>
            <p className="text-xs font-bold">94.2 GB Available</p>
          </div>
          <div className="w-px h-8 bg-slate-200 dark:bg-slate-800" />
          <div className="flex flex-col items-end">
            <p className="text-[10px] font-black uppercase opacity-30 tracking-widest">Latency</p>
            <p className="text-xs font-bold text-primary">12ms</p>
          </div>
        </div>

        <div className="flex items-center gap-2 md:gap-4">
          <button className="hidden sm:flex w-10 h-10 items-center justify-center text-slate-400 hover:text-primary transition-colors">
            <span className="material-symbols-outlined text-2xl">search</span>
          </button>
          <button className="hidden sm:flex w-10 h-10 items-center justify-center text-slate-400 hover:text-primary transition-colors">
            <span className="material-symbols-outlined text-2xl">notifications</span>
          </button>

          <div className="w-9 h-9 md:w-10 md:h-10 rounded-xl md:rounded-2xl bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm flex items-center justify-center overflow-hidden group hover:border-primary/50 transition-all cursor-pointer">
            <span className="material-symbols-outlined text-primary group-hover:scale-110 transition-transform text-lg md:text-xl">person</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default TopHeader;
