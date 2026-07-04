import React from 'react';
import { Search, Library, Network, BarChart3, Settings, History, Plus, Sun, Moon, X } from 'lucide-react';
import Logo from './Logo';

const NAV = [
    { id: 'search', label: 'Search', icon: Search },
    { id: 'library', label: 'Library', icon: Library },
    { id: 'graph', label: 'Graph', icon: Network },
    { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
];

export default function Sidebar({
    activeTab,
    onTabChange,
    onOpenSettings,
    onOpenHistory,
    onNewSearch,
    isOpen,
    onClose,
    darkMode,
    onToggleDark,
}) {
    const handleNav = (id) => {
        onTabChange(id);
        onClose?.();
    };

    return (
        <>
            {/* Mobile backdrop */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-slate-900/50 z-40 lg:hidden"
                    onClick={onClose}
                    aria-hidden="true"
                />
            )}

            <aside
                className={`
                    fixed inset-y-0 left-0 z-50 w-64 flex flex-col
                    bg-white dark:bg-slate-900
                    border-r border-slate-200 dark:border-slate-800
                    transition-transform duration-200
                    ${isOpen ? 'translate-x-0' : '-translate-x-full'}
                    lg:translate-x-0
                `}
            >
                {/* Header */}
                <div className="px-5 pt-5 pb-3 flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                        <Logo size={28} />
                        <div className="leading-tight">
                            <div className="font-bold text-base text-slate-900 dark:text-slate-50">Docu AI</div>
                            <div className="text-[10px] font-medium text-slate-500 dark:text-slate-400">Local document search</div>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="lg:hidden p-1.5 rounded-md text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                        aria-label="Close menu"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* New search button */}
                <div className="px-3 pt-2 pb-1">
                    <button
                        onClick={() => { onNewSearch(); onClose?.(); }}
                        className="w-full inline-flex items-center justify-center gap-2 bg-primary text-white px-3 py-2.5 rounded-lg text-sm font-medium hover:bg-primary/90 active:scale-[0.98] transition"
                    >
                        <Plus className="w-4 h-4" />
                        New Search
                    </button>
                </div>

                {/* Nav */}
                <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-0.5">
                    <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 px-3 py-2">
                        Workspace
                    </div>
                    {NAV.map((item) => {
                        const Icon = item.icon;
                        const active = activeTab === item.id;
                        return (
                            <button
                                key={item.id}
                                onClick={() => handleNav(item.id)}
                                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition ${
                                    active
                                        ? 'bg-primary/10 text-primary font-semibold'
                                        : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                                }`}
                            >
                                <Icon className="w-4 h-4 flex-shrink-0" />
                                <span>{item.label}</span>
                            </button>
                        );
                    })}

                    <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 px-3 py-2 pt-5">
                        Tools
                    </div>
                    <button
                        onClick={() => { onOpenHistory(); onClose?.(); }}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                    >
                        <History className="w-4 h-4" />
                        History
                    </button>
                    <button
                        onClick={() => { onOpenSettings(); onClose?.(); }}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                    >
                        <Settings className="w-4 h-4" />
                        Settings
                    </button>
                </nav>

                {/* Footer */}
                <div className="px-3 py-3 border-t border-slate-200 dark:border-slate-800">
                    <button
                        onClick={onToggleDark}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                    >
                        {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                        {darkMode ? 'Light mode' : 'Dark mode'}
                    </button>
                </div>
            </aside>
        </>
    );
}
