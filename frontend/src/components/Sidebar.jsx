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
                    className="fixed inset-0 bg-black/40 z-40 lg:hidden"
                    onClick={onClose}
                    aria-hidden="true"
                />
            )}

            <aside
                className={`
                    fixed inset-y-0 left-0 z-50 w-64 flex flex-col
                    bg-canvas dark:bg-[#0a0a0a]
                    border-r border-hairline dark:border-[rgba(255,255,255,0.08)]
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
                            <div className="font-semibold text-[15px] text-ink dark:text-[#ededed] tracking-[-0.3px]">Docu AI</div>
                            <div className="text-[11px] font-mono text-mute">search engine</div>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="lg:hidden p-1.5 rounded-v-sm text-mute hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]"
                        aria-label="Close menu"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* New search button */}
                <div className="px-3 pt-2 pb-1">
                    <button
                        onClick={() => { onNewSearch(); onClose?.(); }}
                        className="btn-primary w-full h-9"
                    >
                        <Plus className="w-4 h-4" />
                        New Search
                    </button>
                </div>

                {/* Nav */}
                <nav className="flex-1 overflow-y-auto px-3 py-3 space-y-0.5">
                    <div className="font-mono text-[10px] font-normal uppercase tracking-[0.05em] text-mute dark:text-[#555] px-3 py-2">
                        Workspace
                    </div>
                    {NAV.map((item) => {
                        const Icon = item.icon;
                        const active = activeTab === item.id;
                        return (
                            <button
                                key={item.id}
                                onClick={() => handleNav(item.id)}
                                className={`w-full flex items-center gap-3 px-3 py-2 rounded-v-sm text-sm font-medium transition relative ${
                                    active
                                        ? 'bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.06)] text-ink dark:text-[#ededed]'
                                        : 'text-body dark:text-[#888] hover:bg-canvas-soft dark:hover:bg-[rgba(255,255,255,0.04)]'
                                }`}
                            >
                                {active && (
                                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-4 bg-ink dark:bg-[#ededed] rounded-r-full" />
                                )}
                                <Icon className="w-4 h-4 flex-shrink-0" />
                                <span>{item.label}</span>
                            </button>
                        );
                    })}

                    <div className="font-mono text-[10px] font-normal uppercase tracking-[0.05em] text-mute dark:text-[#555] px-3 py-2 pt-5">
                        Tools
                    </div>
                    <button
                        onClick={() => { onOpenHistory(); onClose?.(); }}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-v-sm text-sm font-medium text-body dark:text-[#888] hover:bg-canvas-soft dark:hover:bg-[rgba(255,255,255,0.04)] transition"
                    >
                        <History className="w-4 h-4" />
                        History
                    </button>
                    <button
                        onClick={() => { onOpenSettings(); onClose?.(); }}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-v-sm text-sm font-medium text-body dark:text-[#888] hover:bg-canvas-soft dark:hover:bg-[rgba(255,255,255,0.04)] transition"
                    >
                        <Settings className="w-4 h-4" />
                        Settings
                    </button>
                </nav>

                {/* Footer */}
                <div className="px-3 py-3 border-t border-hairline dark:border-[rgba(255,255,255,0.08)]">
                    <button
                        onClick={onToggleDark}
                        className="w-full flex items-center gap-3 px-3 py-2 rounded-v-sm text-sm font-medium text-body dark:text-[#888] hover:bg-canvas-soft dark:hover:bg-[rgba(255,255,255,0.04)] transition"
                    >
                        {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                        {darkMode ? 'Light mode' : 'Dark mode'}
                    </button>
                </div>
            </aside>
        </>
    );
}
