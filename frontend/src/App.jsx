import React, { useState, useEffect } from 'react';
import SideNavBar from './components/SideNavBar';
import TopHeader from './components/TopHeader';
import SearchBar from './components/SearchBar';
import SearchResults from './components/SearchResults';
import SearchHistory from './components/SearchHistory';
import SettingsModal from './components/SettingsModal';
import AgentChat from './components/AgentChat';
import AbstractBackground from './components/AbstractBackground';
import FileList from './components/FileList';
import BenchmarkResults from './components/BenchmarkResults';
import ModelComparison from './components/ModelComparison';
import axios from 'axios';
import { AppLogo } from './components/Logo';

function App() {
    const [darkMode, setDarkMode] = useState(true);
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [isHistoryOpen, setIsHistoryOpen] = useState(false);
    const [isSidebarOpen, setIsSidebarOpen] = useState(false);
    const [searchResults, setSearchResults] = useState([]);
    const [aiAnswer, setAiAnswer] = useState("");
    const [activeModel, setActiveModel] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [folders, setFolders] = useState([]);
    const [indexingStatus, setIndexingStatus] = useState({ running: false, progress: 0 });
    const [isAgentMode, setIsAgentMode] = useState(false);
    const [agentQuery, setAgentQuery] = useState("");
    const [systemPrompts, setSystemPrompts] = useState([]);
    const [selectedPromptId, setSelectedPromptId] = useState(null);
    const [activeTab, setActiveTab] = useState('dashboard');
    const [indexedFiles, setIndexedFiles] = useState([]);

    useEffect(() => {
        if (activeTab === 'library') {
            fetchIndexedFiles();
        }
    }, [activeTab]);

    const fetchIndexedFiles = async () => {
        try {
            const res = await axios.get('/api/files');
            setIndexedFiles(res.data || []);
        } catch (error) {
            console.error("Failed to fetch indexed files", error);
        }
    };

    const handleRemoveFile = async (fileId) => {
        try {
            console.log("Removing file", fileId);
        } catch (error) {
            console.error("Failed to remove file", error);
        }
    };

    useEffect(() => {
        let interval;
        const fetchStatus = async () => {
            try {
                const res = await axios.get('/api/index/status');
                setIndexingStatus(res.data);
                if (!res.data.running) clearInterval(interval);
            } catch (e) {
                console.error("Status check failed", e);
            }
        };

        if (indexingStatus.running) {
            interval = setInterval(fetchStatus, 2000);
        }
        return () => clearInterval(interval);
    }, [indexingStatus.running]);

    useEffect(() => {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            setDarkMode(savedTheme === 'dark');
        } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            setDarkMode(true);
        }
        checkConfig();
        fetchModels();
        fetchSystemPrompts();
    }, []);

    // Close sidebar when resizing to desktop
    useEffect(() => {
        const handleResize = () => {
            if (window.innerWidth >= 768) setIsSidebarOpen(false);
        };
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    const fetchSystemPrompts = async () => {
        try {
            const res = await axios.get('/api/system-prompts');
            setSystemPrompts(res.data || []);
        } catch (error) {
            console.error("Failed to fetch system prompts", error);
        }
    };

    const checkConfig = async () => {
        try {
            const res = await axios.get('/api/config');
            setFolders(res.data.folders || []);
            setActiveModel(res.data.active_model);
        } catch (e) {
            console.error("Config check failed", e);
        }
    };

    const fetchModels = async () => {
        try {
            const res = await axios.get('/api/models/local');
            setAvailableModels(res.data);
        } catch (e) {
            console.error("Failed to fetch models", e);
        }
    };

    const handleSearch = async (query) => {
        if (!query.trim()) return;
        setIsLoading(true);
        setHasSearched(true);
        setSearchResults([]);
        setAiAnswer("");
        setAgentQuery(query);

        try {
            const res = await axios.post('/api/search', {
                query,
                provider: 'local',
                system_prompt_id: selectedPromptId
            });

            setSearchResults(res.data.results);
            setIsLoading(false);

            if (res.data.results.length > 0) {
                setAiAnswer("Generating summary...");
                try {
                    const response = await fetch('/api/search/stream', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query,
                            context: res.data.results.map(r => r.content).join("\n\n"),
                            system_prompt_id: selectedPromptId
                        })
                    });

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let accumulatedAnswer = "";
                    let isFirstChunk = true;

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value);
                        if (chunk) {
                            if (isFirstChunk) {
                                accumulatedAnswer = "";
                                isFirstChunk = false;
                            }
                            accumulatedAnswer += chunk;
                            setAiAnswer(accumulatedAnswer);
                        }
                    }
                } catch (streamError) {
                    console.error("Streaming failed:", streamError);
                    setAiAnswer("");
                }
            }
        } catch (error) {
            console.error('Search failed:', error);
            setIsLoading(false);
        }
    };

    const renderTabContent = () => {
        switch (activeTab) {
            case 'dashboard':
                return (
                    <div className={`flex-1 flex flex-col items-center w-full max-w-5xl mx-auto px-4 sm:px-6 ${hasSearched ? 'pt-6 md:pt-8' : 'justify-center min-h-[70vh]'}`}>
                        {!hasSearched && (
                            <div className="text-center space-y-8 md:space-y-10 mb-12 md:mb-16 animate-in fade-in zoom-in duration-1000 px-2">
                                <div className="mx-auto w-20 h-20 md:w-32 md:h-32 rounded-[2rem] md:rounded-[2.5rem] bg-white dark:bg-slate-900 shadow-2xl flex items-center justify-center mb-6 md:mb-8 relative group transition-transform hover:scale-105">
                                    <AppLogo className="w-12 h-12 md:w-20 md:h-20" />
                                    <div className="absolute inset-0 rounded-[2rem] md:rounded-[2.5rem] bg-primary/10 blur-2xl -z-10 group-hover:bg-primary/20 transition-all" />
                                </div>
                                <div className="space-y-3 md:space-y-4">
                                    <h1 className="text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-black text-slate-900 dark:text-white tracking-tighter font-headline">
                                        Neural<span className="text-primary italic">Search</span>
                                    </h1>
                                    <div className="flex items-center justify-center gap-3 md:gap-4">
                                        <div className="h-1 w-8 md:w-12 bg-primary rounded-full" />
                                        <p className="text-slate-500 dark:text-slate-400 text-base sm:text-lg md:text-2xl font-medium opacity-60">
                                            Access your collective intelligence
                                        </p>
                                        <div className="h-1 w-8 md:w-12 bg-primary rounded-full" />
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className={`w-full transition-all duration-700 ${hasSearched ? 'mb-8 md:mb-10' : 'max-w-3xl'}`}>
                            <SearchBar
                                onSearch={handleSearch}
                                isLoading={isLoading}
                                isAgentMode={isAgentMode}
                                onToggleAgent={() => setIsAgentMode(!isAgentMode)}
                                systemPrompts={systemPrompts}
                                selectedPromptId={selectedPromptId}
                                onPromptChange={setSelectedPromptId}
                            />
                        </div>

                        {hasSearched && (
                            <div className="w-full animate-in fade-in slide-in-from-bottom-8 duration-700">
                                {isAgentMode && agentQuery ? (
                                    <AgentChat query={agentQuery} />
                                ) : (
                                    <SearchResults results={searchResults} aiAnswer={aiAnswer} />
                                )}
                            </div>
                        )}
                    </div>
                );
            case 'library':
                return (
                    <div className="max-w-6xl mx-auto py-6 md:py-10 px-1 animate-in fade-in slide-in-from-bottom-8 duration-1000">
                        <div className="mb-8 md:mb-16 flex flex-col sm:flex-row sm:items-end justify-between gap-6">
                            <div>
                                <h2 className="text-3xl md:text-5xl font-black font-headline tracking-tighter mb-2 md:mb-4">Neural Repository</h2>
                                <p className="text-base md:text-xl opacity-60 font-medium max-w-xl">Unified index of your local knowledge assets.</p>
                            </div>
                            <button
                                onClick={() => setIsSettingsOpen(true)}
                                className="self-start sm:self-auto px-5 md:px-8 py-3 md:py-4 rounded-[2rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 font-bold text-xs uppercase tracking-widest hover:shadow-xl transition-all flex items-center gap-3 group flex-shrink-0"
                            >
                                <span className="material-symbols-outlined text-xl group-hover:rotate-180 transition-transform duration-700">settings</span>
                                <span className="hidden sm:inline">Indexer Config</span>
                            </button>
                        </div>
                        <FileList files={indexedFiles} onRemove={handleRemoveFile} />
                    </div>
                );
            case 'workspace':
                return (
                    <div className="max-w-6xl mx-auto py-6 md:py-10 space-y-12 md:space-y-20 px-1 animate-in fade-in slide-in-from-bottom-8 duration-1000">
                        <div className="space-y-3 md:space-y-4">
                            <h2 className="text-3xl md:text-5xl font-black font-headline tracking-tighter">System Telemetry</h2>
                            <p className="text-base md:text-xl opacity-60 font-medium max-w-xl">Performance diagnostics and model behavioral analysis.</p>
                        </div>

                        <div className="space-y-16 md:space-y-24">
                            <section>
                                <BenchmarkResults />
                            </section>
                            <div className="h-px w-full bg-gradient-to-r from-transparent via-[#f3f3fd] dark:via-slate-800 to-transparent" />
                            <section>
                                <ModelComparison />
                            </section>
                        </div>
                    </div>
                );
            default:
                return (
                    <div className="flex flex-col items-center justify-center h-[60vh] text-slate-400 space-y-6">
                        <span className="material-symbols-outlined text-5xl md:text-6xl opacity-20">construction</span>
                        <p className="text-base md:text-xl font-bold font-headline uppercase tracking-widest opacity-40">Module Under Construction</p>
                    </div>
                );
        }
    };

    return (
        <div className="min-h-screen bg-[#faf8ff] dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex font-body selection:bg-primary/20 overflow-hidden">
            <AbstractBackground />

            <SideNavBar
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                onNewSearch={() => { setHasSearched(false); setSearchResults([]); setAiAnswer(""); setActiveTab('dashboard'); }}
                setIsSettingsOpen={setIsSettingsOpen}
                setIsHistoryOpen={setIsHistoryOpen}
                isOpen={isSidebarOpen}
                onClose={() => setIsSidebarOpen(false)}
            />

            {/* Main content — offset for desktop sidebar, full-width on mobile */}
            <div className="flex-1 md:ml-72 flex flex-col relative z-10 h-screen min-w-0">
                <TopHeader
                    title={activeTab}
                    userQuery={agentQuery}
                    onMenuOpen={() => setIsSidebarOpen(true)}
                />

                <main className="flex-1 overflow-y-auto px-4 sm:px-6 md:px-8 lg:px-10 py-6 md:py-8 custom-scrollbar scroll-smooth">
                    {renderTabContent()}
                </main>
            </div>

            {/* Bottom nav bar — mobile only */}
            <nav className="fixed bottom-0 left-0 right-0 md:hidden z-50 bg-[#faf8ff]/95 dark:bg-slate-950/95 backdrop-blur-2xl border-t border-[#f3f3fd] dark:border-slate-800 flex items-center justify-around px-2 py-2 safe-bottom">
                {[
                    { id: 'dashboard', icon: 'search_spark', label: 'Search' },
                    { id: 'library', icon: 'database', label: 'Library' },
                    { id: 'workspace', icon: 'analytics', label: 'Telemetry' },
                ].map(item => (
                    <button
                        key={item.id}
                        onClick={() => setActiveTab(item.id)}
                        className={`flex flex-col items-center gap-1 px-4 py-2 rounded-2xl transition-all duration-200 ${
                            activeTab === item.id
                                ? 'text-primary bg-primary/5'
                                : 'text-slate-400'
                        }`}
                    >
                        <span
                            className="material-symbols-outlined text-2xl"
                            style={activeTab === item.id ? { fontVariationSettings: "'FILL' 1" } : {}}
                        >
                            {item.icon}
                        </span>
                        <span className="text-[10px] font-bold uppercase tracking-wider">{item.label}</span>
                    </button>
                ))}
                <button
                    onClick={() => setIsSettingsOpen(true)}
                    className="flex flex-col items-center gap-1 px-4 py-2 rounded-2xl text-slate-400 transition-all duration-200"
                >
                    <span className="material-symbols-outlined text-2xl">settings</span>
                    <span className="text-[10px] font-bold uppercase tracking-wider">Settings</span>
                </button>
            </nav>

            <SettingsModal
                isOpen={isSettingsOpen}
                onClose={() => setIsSettingsOpen(false)}
                onSave={() => { checkConfig(); fetchModels(); }}
                activeModel={activeModel}
            />

            <SearchHistory
                isOpen={isHistoryOpen}
                onClose={() => setIsHistoryOpen(false)}
                onSelectQuery={handleSearch}
            />
        </div>
    );
}

export default App;
