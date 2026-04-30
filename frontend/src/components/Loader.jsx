import React from 'react';

export const Loader = () => (
    <div className="flex flex-col items-center justify-center gap-6">
        <div className="relative w-24 h-24">
            <div className="absolute inset-0 rounded-[2rem] border-2 border-primary/10 animate-[spin_4s_linear_infinite]"></div>
            <div className="absolute inset-2 rounded-[1.5rem] border-2 border-t-primary animate-spin"></div>
            <div className="absolute inset-6 rounded-[1rem] bg-primary/20 backdrop-blur-xl animate-pulse flex items-center justify-center">
                <span className="material-symbols-outlined text-primary animate-bounce">neurology</span>
            </div>
        </div>
        <p className="text-[10px] font-black uppercase tracking-[0.3em] opacity-40 animate-pulse">Neural Matrix Loading</p>
    </div>
);
