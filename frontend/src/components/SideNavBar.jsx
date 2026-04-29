import React from 'react';
import { AppLogo } from './Logo';

const SideNavBar = ({ activeTab, setActiveTab, onNewSearch, setIsSettingsOpen, setIsHistoryOpen }) => {
  const navItems = [
    { id: 'dashboard', label: 'Neural Search', icon: 'search_spark' },
    { id: 'library', label: 'Knowledge Base', icon: 'database' },
    { id: 'workspace', label: 'Telemetry', icon: 'analytics' },
  ];

  return (
    <aside className="fixed left-0 top-0 h-full p-8 w-72 flex flex-col border-r border-[#f3f3fd] dark:border-slate-800 bg-[#faf8ff] dark:bg-slate-950 font-headline z-50">
      <div className="mb-12 flex items-center gap-3 px-2">
        <AppLogo className="w-12 h-12" />
        <div>
          <h2 className="text-xl font-black tracking-tighter text-slate-900 dark:text-white leading-tight">Docu<span className="text-primary">AI</span></h2>
          <p className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30">Neural Engine v4.0</p>
        </div>
      </div>

      <nav className="flex-1 space-y-3">
        <p className="text-[10px] font-black uppercase tracking-[0.2em] opacity-30 mb-6 ml-4">Command Center</p>
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`w-full flex items-center gap-4 px-6 py-4 transition-all duration-500 rounded-[1.5rem] group relative ${
              activeTab === item.id
                ? 'text-primary dark:text-white font-bold bg-white dark:bg-slate-900 shadow-sm border border-[#f3f3fd] dark:border-slate-800'
                : 'text-slate-400 hover:text-slate-600 dark:hover:text-slate-200'
            }`}
          >
            {activeTab === item.id && (
               <div className="absolute left-0 w-1 h-6 bg-primary rounded-r-full shadow-[2px_0_10px_rgba(0,64,224,0.3)]"></div>
            )}
            <span className={`material-symbols-outlined text-2xl ${activeTab === item.id ? 'fill-current' : 'group-hover:scale-110 transition-transform duration-500'}`}
                  style={activeTab === item.id ? { fontVariationSettings: "'FILL' 1" } : {}}>
              {item.icon}
            </span>
            <span className="text-sm font-bold tracking-tight">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="mt-auto pt-8 space-y-3">
        <button 
          onClick={onNewSearch}
          className="w-full mb-8 bg-primary text-white py-4 rounded-[1.5rem] font-bold flex items-center justify-center gap-3 hover:shadow-2xl hover:shadow-primary/30 transition-all active:scale-95 group"
        >
          <span className="material-symbols-outlined text-xl group-hover:rotate-90 transition-transform duration-500">add</span>
          New Sequence
        </button>
        
        <button 
          onClick={() => setIsHistoryOpen(true)}
          className="w-full flex items-center gap-4 px-6 py-4 text-slate-400 hover:text-primary transition-all rounded-[1.5rem] hover:bg-primary/5 group"
        >
          <span className="material-symbols-outlined text-2xl group-hover:scale-110 transition-transform">history</span>
          <span className="text-sm font-bold tracking-tight">Timeline</span>
        </button>
        
        <button 
          onClick={() => setIsSettingsOpen(true)}
          className="w-full flex items-center gap-4 px-6 py-4 text-slate-400 hover:text-primary transition-all rounded-[1.5rem] hover:bg-primary/5 group"
        >
          <span className="material-symbols-outlined text-2xl group-hover:rotate-45 transition-transform">settings</span>
          <span className="text-sm font-bold tracking-tight">System Core</span>
        </button>

        <div className="mt-8 p-6 rounded-[2rem] bg-[#f3f3fd] dark:bg-slate-900/50 border border-[#f3f3fd] dark:border-slate-800 shadow-sm">
           <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full border-2 border-primary/20 p-0.5">
                 <img 
                   src="https://lh3.googleusercontent.com/aida-public/AB6AXuDQmWahrtP4lY8r3Z7IE2wGtsCHJUf6ut8yHKY7vGoU0NReNVLChr20uLbUS6EDuthaDrbLv1rsOVh9YI8C4RELSsY9GSs0xPZ8rIK2-M31aH9gFplmRoviBhjrmUTBpb30HH9YeKB3LHb8kOxCVQFL9nGwyRQwAYo-MSgmEhTPz5NBWGQsvQzQTOc42BYb5JNhLr2xJpPO2y9dFO2BamRKYlj0ltnLSastgFATWsWNv1v1l-pXIyzDLb2ZGRjPepD_3dAvLpejbgI" 
                   alt="User" 
                   className="w-full h-full object-cover rounded-full"
                 />
              </div>
              <div className="flex-1 min-w-0">
                 <p className="text-sm font-black truncate">Neural Analyst</p>
                 <p className="text-[10px] font-black text-primary uppercase tracking-widest opacity-50">Admin Node</p>
              </div>
           </div>
        </div>
      </div>
    </aside>
  );
};

export default SideNavBar;
