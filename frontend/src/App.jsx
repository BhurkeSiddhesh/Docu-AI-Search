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
    const [searchResults, setSearchResults] = useState([]);
    const [aiAnswer, setAiAnswer] = useState("");
    const [activeModel, setActiveModel] = useState(null);
    const [availableModels, setAvailableModels] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
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
            const res = await axios.get('http://localhost:8000/api/files');
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
        const handleResize = () => setIsMobile(window.innerWidth < 768);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    useEffect(() => {
        let interval;
        const fetchStatus = async () => {
            try {
                const res = await axios.get('http://localhost:8000/api/index/status');
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

    const fetchSystemPrompts = async () => {
        try {
            const res = await axios.get('http://localhost:8000/api/system-prompts');
            setSystemPrompts(res.data || []);
        } catch (error) {
            console.error("Failed to fetch system prompts", error);
        }
    };

    const checkConfig = async () => {
        try {
            const res = await axios.get('http://localhost:8000/api/config');
            setFolders(res.data.folders || []);
            setActiveModel(res.data.active_model);
        } catch (e) {
            console.error("Config check failed", e);
        }
    };

    const fetchModels = async () => {
        try {
            const res = await axios.get('http://localhost:8000/api/models/local');
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
            const res = await axios.post('http://localhost:8000/api/search', {
                query,
                provider: 'local',
                system_prompt_id: selectedPromptId
            });

            setSearchResults(res.data.results);
            setIsLoading(false);

            if (res.data.results.length > 0) {
                // If it's a direct answer, it will be in the summary
                setAiAnswer("Generating summary...");
                try {
                    const response = await fetch('http://localhost:8000/api/search/stream', {
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
                    <div className={`flex-1 flex flex-col items-center w-full max-w-5xl mx-auto px-6 ${hasSearched ? 'pt-8' : 'justify-center min-h-[70vh]'}`}>
                        {!hasSearched && (
                            <div className="text-center space-y-10 mb-16 animate-in fade-in zoom-in duration-1000">
                                <div className="mx-auto w-32 h-32 rounded-[2.5rem] bg-white dark:bg-slate-900 shadow-2xl flex items-center justify-center mb-8 relative group transition-transform hover:scale-105">
                                    <AppLogo className="w-20 h-20" />
                                    <div className="absolute inset-0 rounded-[2.5rem] bg-primary/10 blur-2xl -z-10 group-hover:bg-primary/20 transition-all" />
                                </div>
                                <div className="space-y-4">
                                    <h1 className="text-6xl md:text-8xl font-black text-slate-900 dark:text-white tracking-tighter font-headline">
                                        Neural<span className="text-primary italic">Search</span>
                                    </h1>
                                    <div className="flex items-center justify-center gap-4">
                                        <div className="h-1 w-12 bg-primary rounded-full"></div>
                                        <p className="text-slate-500 dark:text-slate-400 text-xl md:text-2xl font-medium opacity-60">
                                            Access your collective intelligence
                                        </p>
                                        <div className="h-1 w-12 bg-primary rounded-full"></div>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className={`w-full transition-all duration-700 ${hasSearched ? 'mb-10' : 'max-w-3xl'}`}>
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
                    <div className="max-w-6xl mx-auto py-10 animate-in fade-in slide-in-from-bottom-8 duration-1000">
                        <div className="mb-16 flex items-end justify-between">
                            <div>
                                <h2 className="text-5xl font-black font-headline tracking-tighter mb-4">Neural Repository</h2>
                                <p className="text-xl opacity-60 font-medium max-w-xl">Unified index of your local knowledge assets. Encrypted and optimized for rapid retrieval.</p>
                            </div>
                            <button 
                                onClick={() => setIsSettingsOpen(true)}
                                className="px-8 py-4 rounded-[2rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 font-bold text-xs uppercase tracking-widest hover:shadow-xl transition-all flex items-center gap-3 group"
                            >
                                <span className="material-symbols-outlined text-xl group-hover:rotate-180 transition-transform duration-700">settings</span>
                                Indexer Configuration
                            </button>
                        </div>
                        <FileList files={indexedFiles} onRemove={handleRemoveFile} />
                    </div>
                );
            case 'workspace':
                return (
                    <div className="max-w-6xl mx-auto py-10 space-y-20 animate-in fade-in slide-in-from-bottom-8 duration-1000">
                        <div className="space-y-4">
                            <h2 className="text-5xl font-black font-headline tracking-tighter">System Telemetry</h2>
                            <p className="text-xl opacity-60 font-medium max-w-xl">Performance diagnostics and model behavioral analysis.</p>
                        </div>
                        
                        <div className="space-y-24">
                            <section>
                                <BenchmarkResults />
                            </section>
                            
                            <div className="h-px w-full bg-gradient-to-r from-transparent via-[#f3f3fd] dark:via-slate-800 to-transparent"></div>
                            
                            <section>
                                <ModelComparison />
                            </section>
                        </div>
                    </div>
                );
            default:
                return (
                    <div className="flex flex-col items-center justify-center h-[60vh] text-slate-400 space-y-6">
                        <span className="material-symbols-outlined text-6xl opacity-20">construction</span>
                        <p className="text-xl font-bold font-headline uppercase tracking-widest opacity-40">Module Under Construction</p>
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
            />

            <div className="flex-1 ml-72 flex flex-col relative z-10 h-screen">
                <TopHeader 
                    title={activeTab} 
                    userQuery={agentQuery}
                />
                
                <main className="flex-1 overflow-y-auto p-10 custom-scrollbar scroll-smooth">
                    {renderTabContent()}
                </main>
            </div>

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
                isMobile={isMobile}
            />
        </div>
    );
}

export default App;