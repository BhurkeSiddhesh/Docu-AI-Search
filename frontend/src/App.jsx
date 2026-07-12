import React, { useEffect, useState } from 'react';
import { Menu, Search as SearchIcon } from 'lucide-react';
import Sidebar from './components/Sidebar';
import SearchView from './components/SearchView';
import LibraryView from './components/LibraryView';
import GraphView from './components/GraphView';
import BenchmarkView from './components/BenchmarkView';
import SettingsModal from './components/SettingsModal';
import HistoryDrawer from './components/HistoryDrawer';
import IndexingBanner from './components/IndexingBanner';
import ErrorBoundary from './components/ErrorBoundary';
import api from './lib/api';

export default function App() {
    const [darkMode, setDarkMode] = useState(() => {
        const stored = localStorage.getItem('docu-ai-theme');
        if (stored) return stored === 'dark';
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    });
    const [activeTab, setActiveTab] = useState('search');
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [historyOpen, setHistoryOpen] = useState(false);
    const [pendingQuery, setPendingQuery] = useState(null);
    const [resetKey, setResetKey] = useState(0);

    useEffect(() => {
        const root = document.documentElement;
        if (darkMode) root.classList.add('dark');
        else root.classList.remove('dark');
        localStorage.setItem('docu-ai-theme', darkMode ? 'dark' : 'light');
    }, [darkMode]);

    // On startup, fetch the auth token from the backend if AUTH_ENABLED and not yet stored
    useEffect(() => {
        if (localStorage.getItem('api_token')) return;
        api.getAuthToken().then((r) => {
            if (r.data?.token) localStorage.setItem('api_token', r.data.token);
        }).catch(() => { /* auth disabled or already retrieved */ });
    }, []);

    const handleSelectQuery = (q) => {
        setActiveTab('search');
        setPendingQuery({ q, k: Date.now() });
    };

    const handleNewSearch = () => {
        setActiveTab('search');
        setPendingQuery(null);
        setResetKey((k) => k + 1);
    };

    return (
    return (
        <div className={`h-screen supports-[height:100dvh]:h-dvh overflow-hidden ${darkMode ? 'bg-[#0a0a0a]' : 'bg-[#fafafa]'}`}>
            <Sidebar
                activeTab={activeTab}
                onTabChange={setActiveTab}
                onOpenSettings={() => setSettingsOpen(true)}
                onOpenHistory={() => setHistoryOpen(true)}
                onNewSearch={handleNewSearch}
                isOpen={sidebarOpen}
                onClose={() => setSidebarOpen(false)}
                darkMode={darkMode}
                onToggleDark={() => setDarkMode((v) => !v)}
            />

            <div className="lg:ml-64 h-full flex flex-col">
                {/* Top bar — mobile */}
                <header className="lg:hidden flex items-center justify-between gap-3 px-4 h-16 border-b border-hairline dark:border-[rgba(255,255,255,0.1)] bg-canvas dark:bg-[#171717] sticky top-0 z-30">
                    <button
                        onClick={() => setSidebarOpen(true)}
                        className="p-2 rounded-v-sm hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]"
                        aria-label="Open menu"
                    >
                        <Menu className="w-5 h-5 text-ink dark:text-[#ededed]" />
                    </button>
                    <div className="flex items-center gap-2 text-sm font-semibold text-ink dark:text-[#ededed]">
                        <SearchIcon className="w-4 h-4" />
                        Docu AI
                    </div>
                    <div className="w-9" />
                </header>

                {/* Global indexing banner */}
                <div className="px-4 sm:px-6 pt-4">
                    <div className="max-w-5xl mx-auto">
                        <IndexingBanner />
                    </div>
                </div>

                <main className={`flex-1 min-w-0 min-h-0 ${activeTab === 'search' ? 'overflow-hidden' : 'overflow-y-auto'}`}>
                    <ErrorBoundary>
                        {activeTab === 'search' && (
                            <SearchView
                                key={resetKey}
                                pendingQuery={pendingQuery}
                            />
                        </ErrorBoundary>
                    )}
                    {activeTab === 'library' && (
                        <ErrorBoundary>
                            <LibraryView onOpenSettings={() => setSettingsOpen(true)} />
                        )}
                        {activeTab === 'graph' && <GraphView />}
                        {activeTab === 'benchmarks' && <BenchmarkView />}
                    </ErrorBoundary>
                </main>
            </div>

            <SettingsModal
                isOpen={settingsOpen}
                onClose={() => setSettingsOpen(false)}
                onSaved={() => { /* config reloads itself on next open */ }}
            />

            <HistoryDrawer
                isOpen={historyOpen}
                onClose={() => setHistoryOpen(false)}
                onSelectQuery={handleSelectQuery}
            />
        </div>
    );
}
