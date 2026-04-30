import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import ModelManager from './ModelManager';

const API = 'http://localhost:8000';

const Toast = ({ message, type = 'success', onDismiss }) => {
    useEffect(() => {
        const t = setTimeout(onDismiss, 3000);
        return () => clearTimeout(t);
    }, [onDismiss]);

    const colours = {
        success: 'bg-green-500 text-white shadow-green-500/20',
        error:   'bg-red-500 text-white shadow-red-500/20',
        info:    'bg-primary text-white shadow-primary/20',
    };

    return (
        <div className={`fixed bottom-8 right-8 z-[200] flex items-center gap-4 px-6 py-4 rounded-3xl shadow-2xl backdrop-blur-xl text-sm font-bold animate-in slide-in-from-bottom-8 duration-500 ${colours[type]}`}>
            <span className="material-symbols-outlined">{type === 'success' ? 'check_circle' : (type === 'error' ? 'error' : 'info')}</span>
            <span>{message}</span>
            <button type="button" onClick={onDismiss} className="ml-2 opacity-70 hover:opacity-100">
                <span className="material-symbols-outlined text-sm">close</span>
            </button>
        </div>
    );
};

const EMBEDDING_PROVIDER_TYPES = [
    { value: 'local',           label: 'Local (On-device)',       needsKey: false },
    { value: 'huggingface_api', label: 'HuggingFace API',           needsKey: true  },
    { value: 'commercial_api',  label: 'Cloud (OpenAI/Gemini)',    needsKey: true  },
];

const DEFAULT_EMBEDDING_CONFIG = {
    provider_type: 'local',
    model_name:    'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    api_key:       '',
};

const SettingsModal = ({ isOpen, onClose, onSave, activeModel }) => {
    const [config, setConfig] = useState({
        folders: [],
        auto_index: false,
        provider: 'local',
        openai_api_key: '',
        gemini_api_key: '',
        anthropic_api_key: '',
        grok_api_key: '',
        local_model_path: '',
        local_model_type: 'llamacpp'
    });
    const [embeddingConfig, setEmbeddingConfig] = useState(DEFAULT_EMBEDDING_CONFIG);
    const [toast, setToast] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [indexingStatus, setIndexingStatus] = useState({ running: false, progress: 0 });
    const [activeSection, setActiveSection] = useState('folders');
    const [folderHistory, setFolderHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);
    const [cacheStats, setCacheStats] = useState({ total_entries: 0, total_hits: 0 });
    const [extProviderType, setExtProviderType] = useState('lmstudio');
    const [extHealth, setExtHealth] = useState(null);
    const [extModels, setExtModels] = useState([]);
    const [extLoadingHealth, setExtLoadingHealth] = useState(false);
    const [extLoadingModels, setExtLoadingModels] = useState(false);

    const showToast = useCallback((message, type = 'success') => {
        setToast({ message, type });
    }, []);
    const dismissToast = useCallback(() => setToast(null), []);

    useEffect(() => {
        if (isOpen) {
            fetchConfig();
            fetchEmbeddingConfig();
            fetchFolderHistory();
            fetchCacheStats();
        }
    }, [isOpen]);

    useEffect(() => {
        let interval;
        if (isOpen) {
            fetchIndexingStatus();
            interval = setInterval(fetchIndexingStatus, 1500);
        }
        return () => clearInterval(interval);
    }, [isOpen]);

    const fetchConfig = async () => {
        try {
            const response = await axios.get(`${API}/api/config`);
            const data = response.data;
            const normalized = { ...data };
            ['openai', 'gemini', 'anthropic', 'grok'].forEach(p => {
                normalized[`${p}_api_key`] = '';
            });
            setConfig(prev => ({ ...prev, ...normalized }));
        } catch (error) {
            console.error('Failed to fetch config:', error);
        }
    };

    const fetchEmbeddingConfig = async () => {
        try {
            const response = await axios.get(`${API}/api/settings/embeddings`);
            setEmbeddingConfig({
                provider_type: response.data.provider_type,
                model_name:    response.data.model_name,
                api_key: response.data.api_key_set ? '••••••••' : '',
            });
        } catch (error) {
            console.error('Failed to fetch embedding config:', error);
        }
    };

    const fetchIndexingStatus = async () => {
        try {
            const response = await axios.get(`${API}/api/index/status`);
            setIndexingStatus(response.data);
        } catch (error) {
            console.error('Failed to fetch status:', error);
        }
    };

    const fetchFolderHistory = async () => {
        try {
            const response = await axios.get(`${API}/api/folders/history`);
            setFolderHistory(response.data || []);
        } catch (error) {
            console.error('Failed to fetch folder history:', error);
        }
    };

    const fetchCacheStats = async () => {
        try {
            const response = await axios.get(`${API}/api/cache/stats`);
            setCacheStats(response.data);
        } catch (error) {
            console.error('Failed to fetch cache stats:', error);
        }
    };

    const handleSave = async () => {
        setIsLoading(true);
        try {
            await axios.post(`${API}/api/config`, config);
            const embPayload = {
                provider_type: embeddingConfig.provider_type,
                model_name:    embeddingConfig.model_name,
            };
            if (embeddingConfig.api_key !== '••••••••' && embeddingConfig.api_key) {
                embPayload.api_key = embeddingConfig.api_key;
            }
            await axios.post(`${API}/api/settings/embeddings`, embPayload);
            showToast('Configuration updated!');
            onSave();
            onClose();
        } catch (error) {
            showToast(error.response?.data?.detail || 'Update failed', 'error');
        } finally {
            setIsLoading(false);
        }
    };

    const handleAddFolder = async () => {
        try {
            const response = await axios.get(`${API}/api/browse`);
            if (response.data.folder && !config.folders.includes(response.data.folder)) {
                const newFolders = [...config.folders, response.data.folder];
                setConfig(prev => ({ ...prev, folders: newFolders }));
                await axios.post(`${API}/api/config`, { ...config, folders: newFolders });
                fetchFolderHistory();
            }
        } catch (error) {
            console.error('Failed to browse:', error);
        }
    };

    if (!isOpen) return null;

    const sections = [
        { id: 'folders', label: 'Library', icon: 'folder_open' },
        { id: 'providers', label: 'Cloud AI', icon: 'cloud' },
        { id: 'external', label: 'External', icon: 'dns' },
        { id: 'embeddings', label: 'Embeddings', icon: 'database' },
        { id: 'local', label: 'Local LLM', icon: 'memory' },
        { id: 'data', label: 'System', icon: 'settings' },
    ];

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm" onClick={(e) => e.target === e.currentTarget && onClose()} onKeyDown={(e) => e.key === 'Escape' && onClose()} tabIndex={-1} aria-hidden="true">
            <div className="glass-overlay w-[96vw] max-w-6xl h-[88vh] rounded-[2.5rem] shadow-2xl overflow-hidden flex flex-col" role="dialog" aria-modal="true" aria-labelledby="settings-modal-title">
                {/* Header */}
                <div className="px-8 py-6 border-b border-border/30 bg-background/85 backdrop-blur-md flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary">
                            <span className="material-symbols-outlined text-2xl">settings</span>
                        </div>
                        <div>
                            <h2 id="settings-modal-title" className="text-xl font-bold font-headline text-[#191b22] dark:text-white uppercase tracking-tight">System Configuration</h2>
                            <p className="text-[10px] font-black opacity-40 uppercase tracking-widest">Adjust your AI workspace parameters</p>
                        </div>
                    </div>
                    <button onClick={onClose} aria-label="Close settings" className="w-12 h-12 rounded-full hover:bg-secondary/50 flex items-center justify-center transition-all">
                        <span className="material-symbols-outlined">close</span>
                    </button>
                </div>

                <div className="flex-1 flex overflow-hidden">
                    {/* Sidebar */}
                    <aside className="w-72 bg-[#f3f3fd] dark:bg-slate-950/40 p-6 space-y-2 border-r border-[#f3f3fd] dark:border-slate-800">
                        {sections.map(s => (
                            <button
                                key={s.id}
                                onClick={() => setActiveSection(s.id)}
                                className={`w-full flex items-center gap-3 px-5 py-4 rounded-2xl transition-all font-headline ${activeSection === s.id ? 'bg-white dark:bg-slate-900 text-primary shadow-sm font-bold scale-105' : 'text-[#434656] dark:text-slate-400 hover:bg-white/50 dark:hover:bg-slate-900/50'}`}
                            >
                                <span className={`material-symbols-outlined ${activeSection === s.id ? 'fill-current' : ''}`}>{s.icon}</span>
                                <span className="text-sm">{s.label}</span>
                            </button>
                        ))}
                    </aside>

                    {/* Content */}
                    <main className="flex-1 overflow-y-auto p-10 custom-scrollbar space-y-12">
                        {!config ? (
                            <div className="flex items-center justify-center h-full">
                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                            </div>
                        ) : (
                            <>
                                {activeSection === 'folders' && (
                            <div className="space-y-8 animate-in fade-in slide-in-from-right-4 duration-500">
                                <div className="space-y-6">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3 px-2">
                                            <h3 className="text-2xl font-bold font-headline">Knowledge Library</h3>
                                            <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                                            <span className="text-[10px] font-black uppercase tracking-tighter opacity-40">{config.folders.length} Connected</span>
                                        </div>
                                        <button 
                                            onClick={() => setShowHistory(!showHistory)} 
                                            aria-label="Recent History"
                                            className={`p-3 rounded-2xl transition-all ${showHistory ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'bg-white dark:bg-slate-900 text-primary hover:bg-primary/5'}`}
                                        >
                                            <span className="material-symbols-outlined text-lg">history</span>
                                        </button>
                                    </div>

                                    <div className="flex gap-4">
                                        <button 
                                            onClick={async () => {
                                                try {
                                                    await axios.post(`${API}/api/index`);
                                                    showToast('Index rebuild started', 'info');
                                                } catch (e) {
                                                    showToast('Failed to start indexing', 'error');
                                                }
                                            }}
                                            className="flex-1 bg-white dark:bg-slate-900 border border-[#d1d1f0] dark:border-slate-800 p-4 rounded-3xl font-bold text-xs flex items-center justify-center gap-3 hover:border-primary/40 hover:bg-primary/5 transition-all group"
                                        >
                                            <span className="material-symbols-outlined text-primary group-hover:rotate-180 transition-transform duration-500">sync</span>
                                            Rebuild Index
                                        </button>
                                        <button onClick={handleAddFolder} className="bg-primary text-white px-6 py-3 rounded-2xl font-bold flex items-center gap-2 hover:shadow-lg hover:shadow-primary/20 transition-all active:scale-95">
                                            <span className="material-symbols-outlined text-sm">add</span>
                                            Add Folder
                                        </button>
                                    </div>

                                    {showHistory && (
                                        <div className="bg-[#f3f3fd] dark:bg-slate-950/40 rounded-3xl p-6 border border-[#d1d1f0] dark:border-slate-800 space-y-4 animate-in slide-in-from-top-4 duration-300">
                                            <div className="flex items-center justify-between px-2">
                                                <h4 className="text-xs font-black uppercase tracking-widest opacity-40">Previously Indexed</h4>
                                                <button 
                                                    onClick={async () => {
                                                        if (confirm('Clear folder history?')) {
                                                            await axios.delete(`${API}/api/folders/history`);
                                                            fetchFolderHistory();
                                                        }
                                                    }}
                                                    className="text-[10px] font-black uppercase tracking-widest text-red-500 hover:underline"
                                                >
                                                    Clear All
                                                </button>
                                            </div>
                                            <div className="space-y-2">
                                                {folderHistory.length === 0 ? (
                                                    <p className="text-xs opacity-40 px-2">No indexed folders yet.</p>
                                                ) : (
                                                    folderHistory.map(h => (
                                                        <div key={h} className="flex items-center justify-between p-3 rounded-2xl hover:bg-white dark:hover:bg-slate-900 transition-all group">
                                                            <span className="text-xs font-medium truncate flex-1">{h}</span>
                                                            <button 
                                                                onClick={async () => {
                                                                    if (!config.folders.includes(h)) {
                                                                        const next = [...config.folders, h];
                                                                        setConfig(prev => ({ ...prev, folders: next }));
                                                                        await axios.post(`${API}/api/config`, { ...config, folders: next });
                                                                    }
                                                                    setShowHistory(false);
                                                                }}
                                                                className="px-3 py-1.5 rounded-xl bg-primary/10 text-primary text-[10px] font-black uppercase opacity-0 group-hover:opacity-100 transition-all"
                                                            >
                                                                Restore
                                                            </button>
                                                        </div>
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </div>
                                
                                <div className="space-y-4">
                                    {config.folders.length === 0 ? (
                                        <div className="p-12 rounded-[2.5rem] bg-[#f3f3fd] dark:bg-slate-950/40 border-2 border-dashed border-[#d1d1f0] dark:border-slate-800 flex flex-col items-center text-center">
                                            <span className="material-symbols-outlined text-5xl text-[#d1d1f0] mb-4">folder_off</span>
                                            <p className="font-bold text-[#434656] opacity-60">No folders connected to AI Index</p>
                                        </div>
                                    ) : (
                                        config.folders.map(f => (
                                            <div key={f} className="bg-[#f3f3fd] dark:bg-slate-800/40 p-5 rounded-3xl flex items-center justify-between group">
                                                <div className="flex items-center gap-4 min-w-0">
                                                    <span className="material-symbols-outlined text-primary">folder</span>
                                                    <span className="font-bold truncate text-sm">{f}</span>
                                                </div>
                                                <button 
                                                    onClick={() => setConfig(prev => ({ ...prev, folders: prev.folders.filter(x => x !== f) }))}
                                                    aria-label={`Remove ${f} from index`}
                                                    className="w-10 h-10 rounded-full hover:bg-red-500/10 text-red-500 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all"
                                                >
                                                    <span className="material-symbols-outlined text-lg">delete</span>
                                                </button>
                                            </div>
                                        ))
                                    )}
                                </div>

                                {indexingStatus.running && (
                                    <div className="bg-primary/5 border border-primary/10 p-8 rounded-[2.5rem] space-y-4">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-3">
                                                <span className="material-symbols-outlined animate-spin text-primary">progress_activity</span>
                                                <span className="font-bold">Neural Indexing in Progress</span>
                                            </div>
                                            <span className="font-black text-primary">{indexingStatus.progress}%</span>
                                        </div>
                                        <div className="h-3 w-full bg-primary/10 rounded-full overflow-hidden">
                                            <div className="h-full bg-primary transition-all duration-500" style={{ width: `${indexingStatus.progress}%` }}></div>
                                        </div>
                                        <p className="text-[10px] font-black uppercase tracking-widest opacity-40 truncate">{indexingStatus.current_file}</p>
                                    </div>
                                )}
                            </div>
                        )}

                        {activeSection === 'providers' && (
                            <div className="space-y-10 animate-in fade-in slide-in-from-right-4 duration-500">
                                <h3 className="text-2xl font-bold font-headline">Cloud Intelligence</h3>
                                <div className="grid grid-cols-1 gap-6">
                                    {[
                                        { id: 'openai', name: 'OpenAI (GPT-4)', key: 'openai_api_key' },
                                        { id: 'gemini', name: 'Google (Gemini Pro)', key: 'gemini_api_key' },
                                        { id: 'anthropic', name: 'Anthropic (Claude)', key: 'anthropic_api_key' },
                                        { id: 'grok', name: 'xAI (Grok)', key: 'grok_api_key' },
                                    ].map(p => (
                                        <div key={p.id} className="space-y-3">
                                            <label htmlFor={p.key} className="text-xs font-black uppercase tracking-widest opacity-40 px-2">{p.name}</label>
                                            <input 
                                                id={p.key}
                                                type="password"
                                                className="w-full bg-[#f3f3fd] dark:bg-slate-950/40 p-5 rounded-3xl border-2 border-transparent focus:border-primary/20 outline-none transition-all font-body text-sm"
                                                placeholder="Enter API Key..."
                                                value={config[p.key] || ''}
                                                onChange={(e) => setConfig({ ...config, [p.key]: e.target.value })}
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {activeSection === 'local' && (
                            <div className="animate-in fade-in slide-in-from-right-4 duration-500">
                                <ModelManager onModelSelect={(m) => setConfig(prev => ({ ...prev, local_model_path: m.path }))} activeModelPath={config.local_model_path} />
                            </div>
                        )}

                        {activeSection === 'embeddings' && (
                            <div className="space-y-10 animate-in fade-in slide-in-from-right-4 duration-500">
                                <h3 className="text-2xl font-bold font-headline">Neural Embeddings</h3>
                                <div className="space-y-6">
                                    <div className="grid grid-cols-3 gap-4">
                                        {EMBEDDING_PROVIDER_TYPES.map(t => (
                                            <button
                                                key={t.value}
                                                onClick={() => setEmbeddingConfig(prev => ({ ...prev, provider_type: t.value }))}
                                                className={`p-6 rounded-[2rem] text-left transition-all border-2 ${embeddingConfig.provider_type === t.value ? 'bg-primary/5 border-primary text-primary shadow-lg shadow-primary/5' : 'bg-[#f3f3fd] dark:bg-slate-950/40 border-transparent hover:bg-[#ebebfa]'}`}
                                            >
                                                <span className="material-symbols-outlined mb-3 block">hub</span>
                                                <span className="font-bold text-sm leading-tight block">{t.label}</span>
                                            </button>
                                        ))}
                                    </div>

                                    <div className="space-y-3 pt-4">
                                        <label htmlFor="emb-model-name" className="text-xs font-black uppercase tracking-widest opacity-40 px-2">Model Architecture</label>
                                        <input 
                                            id="emb-model-name"
                                            type="text"
                                            className="w-full bg-[#f3f3fd] dark:bg-slate-950/40 p-5 rounded-3xl border-2 border-transparent focus:border-primary/20 outline-none transition-all font-mono text-xs"
                                            value={embeddingConfig.model_name}
                                            onChange={(e) => setEmbeddingConfig(prev => ({ ...prev, model_name: e.target.value }))}
                                        />
                                    </div>
                                    
                                    {EMBEDDING_PROVIDER_TYPES.find(x => x.value === embeddingConfig.provider_type)?.needsKey && (
                                        <div className="space-y-3">
                                            <label htmlFor="emb-api-key" className="text-xs font-black uppercase tracking-widest opacity-40 px-2">Embedding API Key</label>
                                            <input 
                                                id="emb-api-key"
                                                type="password"
                                                className="w-full bg-[#f3f3fd] dark:bg-slate-950/40 p-5 rounded-3xl border-2 border-transparent focus:border-primary/20 outline-none transition-all font-body text-sm"
                                                value={embeddingConfig.api_key}
                                                onChange={(e) => setEmbeddingConfig(prev => ({ ...prev, api_key: e.target.value }))}
                                            />
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                        
                        {activeSection === 'data' && (
                            <div className="space-y-10 animate-in fade-in slide-in-from-right-4 duration-500">
                                <h3 className="text-2xl font-bold font-headline">System Hygiene</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="bg-[#f3f3fd] dark:bg-slate-950/40 p-8 rounded-[2.5rem] space-y-6">
                                        <div>
                                            <h4 className="font-bold text-lg mb-1 text-primary">Response Cache</h4>
                                            <p className="text-xs font-medium opacity-60 leading-relaxed">Stored patterns for instant retrieval of complex answers.</p>
                                        </div>
                                        <div className="flex items-center gap-8">
                                            <div>
                                                <p className="text-2xl font-black">{cacheStats.total_entries}</p>
                                                <p className="text-[10px] font-black uppercase opacity-40">Entries</p>
                                            </div>
                                            <div className="h-8 w-px bg-[#d1d1f0] dark:bg-slate-800"></div>
                                            <div>
                                                <p className="text-2xl font-black text-primary">{cacheStats.total_hits}</p>
                                                <p className="text-[10px] font-black uppercase opacity-40">Total Hits</p>
                                            </div>
                                        </div>
                                        <button onClick={async () => { await axios.post(`${API}/api/cache/clear`); fetchCacheStats(); showToast('Cache Purged'); }} className="w-full py-4 rounded-2xl bg-white dark:bg-slate-900 font-bold text-xs hover:bg-red-500 hover:text-white transition-all shadow-sm">
                                            Purge AI Cache
                                        </button>
                                    </div>

                                    <div className="bg-red-500/5 p-8 rounded-[2.5rem] space-y-6 border border-red-500/10">
                                        <div>
                                            <h4 className="font-bold text-lg mb-1 text-red-500 text-on-red">Privacy Reset</h4>
                                            <p className="text-xs font-medium opacity-60 leading-relaxed text-on-red">Wipe all search queries and local indexing history.</p>
                                        </div>
                                        <button onClick={async () => { if(confirm('Wipe history?')) { await axios.delete(`${API}/api/search/history`); showToast('History Cleared'); } }} className="w-full py-4 rounded-2xl bg-red-500 text-white font-bold text-xs hover:bg-red-600 transition-all shadow-lg shadow-red-500/20">
                                            Clear All History
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )}
                    </main>
                </div>

                {/* Footer */}
                <div className="px-8 py-6 bg-[#f3f3fd] dark:bg-slate-950/60 border-t border-[#f3f3fd] dark:border-slate-800 flex items-center justify-between">
                    <p className="text-xs font-bold opacity-40">v2.4.0 • Neural Search Engine</p>
                    <div className="flex items-center gap-4">
                        <button onClick={onClose} className="px-8 py-3 rounded-2xl font-bold text-sm hover:bg-[#ebebfa] dark:hover:bg-slate-800 transition-all">Cancel</button>
                        <button onClick={handleSave} disabled={isLoading} className="px-10 py-3 rounded-2xl bg-primary text-white font-bold text-sm hover:shadow-xl hover:shadow-primary/20 transition-all active:scale-95 disabled:opacity-50">
                            {isLoading ? 'Saving...' : 'Apply Changes'}
                        </button>
                    </div>
                </div>
            </div>
            {toast && <Toast message={toast.message} type={toast.type} onDismiss={dismissToast} />}
        </div>
    );
};

export default SettingsModal;
