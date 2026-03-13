import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { X, Settings, FolderOpen, Loader2, Save, Key, Cpu, Trash2, CheckCircle2, Database } from 'lucide-react';
import ModelManager from './ModelManager';

const API = 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Inline toast — no extra npm package needed
// ---------------------------------------------------------------------------
const Toast = ({ message, type = 'success', onDismiss }) => {
    useEffect(() => {
        const t = setTimeout(onDismiss, 3000);
        return () => clearTimeout(t);
    }, [onDismiss]);

    const colours = {
        success: 'bg-green-500/90 text-white border-green-400/30',
        error:   'bg-destructive/90 text-white border-destructive/30',
        info:    'bg-primary/90 text-white border-primary/30',
    };

    return (
        <div
            id="settings-toast"
            className={`fixed bottom-6 right-6 z-[200] flex items-center gap-3 px-4 py-3 rounded-xl shadow-2xl border backdrop-blur-md text-sm font-medium animate-in slide-in-from-bottom-4 duration-300 ${colours[type]}`}
        >
            {type === 'success' && <CheckCircle2 className="w-4 h-4 flex-shrink-0" />}
            {type === 'error'   && <X            className="w-4 h-4 flex-shrink-0" />}
            <span>{message}</span>
            <button type="button" onClick={onDismiss} className="ml-2 opacity-70 hover:opacity-100" aria-label="Dismiss notification">
                <X className="w-3 h-3" />
            </button>
        </div>
    );
};

const EMBEDDING_PROVIDER_TYPES = [
    { value: 'local',           label: 'Local (HuggingFace on-device)',       needsKey: false },
    { value: 'huggingface_api', label: 'HuggingFace Inference API',           needsKey: true  },
    { value: 'commercial_api',  label: 'Commercial API (OpenAI / Gemini)',    needsKey: true  },
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
        // API Keys for different providers
        openai_api_key: '',
        gemini_api_key: '',
        anthropic_api_key: '',
        grok_api_key: '',
        // Local model settings
        local_model_path: '',
        local_model_type: 'llamacpp'
    });
    const [embeddingConfig, setEmbeddingConfig] = useState(DEFAULT_EMBEDDING_CONFIG);
    const [toast, setToast] = useState(null);  // { message, type }
    const [isLoading, setIsLoading] = useState(false);
    const [indexingStatus, setIndexingStatus] = useState({
        running: false,
        progress: 0,
        current_file: '',
        total_files: 0,
        processed_files: 0
    });
    const [activeSection, setActiveSection] = useState('folders');
    const [folderHistory, setFolderHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);
    const [cacheStats, setCacheStats] = useState({ total_entries: 0, total_hits: 0 });

    const showToast = useCallback((message, type = 'success') => {
        setToast({ message, type });
    }, []);
    const dismissToast = useCallback(() => setToast(null), []);

    useEffect(() => {
        if (isOpen) {
            fetchConfig();
            fetchEmbeddingConfig();
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

    useEffect(() => {
        if (isOpen) {
            fetchFolderHistory();
            fetchCacheStats();
        }
    }, [isOpen]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape' && isOpen) {
                onClose();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen, onClose]);

    const fetchConfig = async () => {
        try {
            const response = await axios.get(`${API}/api/config`);
            setConfig(prev => ({ ...prev, ...response.data }));
        } catch (error) {
            console.error('Failed to fetch config:', error);
        }
    };

    const fetchEmbeddingConfig = async () => {
        try {
            const response = await axios.get(`${API}/api/settings/embeddings`);
            // Map api_key_set back to empty string for the input field
            setEmbeddingConfig(prev => ({
                ...DEFAULT_EMBEDDING_CONFIG,
                provider_type: response.data.provider_type,
                model_name:    response.data.model_name,
                // If key was set server-side, show placeholder; user can overwrite
                api_key: response.data.api_key_set ? '••••••••' : '',
            }));
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
            // 1. Save general config
            await axios.post(`${API}/api/config`, config);

            // 2. Save embedding config (only send api_key if user typed a new value)
            const embPayload = {
                provider_type: embeddingConfig.provider_type,
                model_name:    embeddingConfig.model_name,
            };
            const keyIsPlaceholder = embeddingConfig.api_key === '••••••••';
            if (!keyIsPlaceholder && embeddingConfig.api_key) {
                embPayload.api_key = embeddingConfig.api_key;
            }
            await axios.post(`${API}/api/settings/embeddings`, embPayload);

            showToast('Settings saved successfully!', 'success');
            onSave();
            onClose();
        } catch (error) {
            console.error('Failed to save config:', error);
            const detail = error.response?.data?.detail || 'Failed to save settings.';
            showToast(detail, 'error');
        } finally {
            setIsLoading(false);
        }
    };

    const saveConfigData = async (newConfig) => {
        try {
            await axios.post(`${API}/api/config`, newConfig);
            // Refresh history after save
            fetchFolderHistory();
        } catch (error) {
            console.error('Failed to save config:', error);
        }
    };

    const handleAddFolder = async () => {
        try {
            const response = await axios.get(`${API}/api/browse`);
            if (response.data.folder && !config.folders.includes(response.data.folder)) {
                const newConfig = {
                    ...config,
                    folders: [...config.folders, response.data.folder]
                };
                setConfig(newConfig);
                await saveConfigData(newConfig);
            }
        } catch (error) {
            console.error('Failed to browse:', error);
        }
    };

    const handleAddFromHistory = async (folder) => {
        if (!config.folders.includes(folder)) {
            const newConfig = {
                ...config,
                folders: [...config.folders, folder]
            };
            setConfig(newConfig);
            await saveConfigData(newConfig);
            setShowHistory(false);
        }
    };

    const handleRemoveFolder = async (folderToRemove) => {
        const newConfig = {
            ...config,
            folders: config.folders.filter(f => f !== folderToRemove)
        };
        setConfig(newConfig);
        await saveConfigData(newConfig);
    };

    const handleRemoveFromHistory = async (folder, e) => {
        e.stopPropagation();  // Prevent adding the folder when clicking delete
        try {
            await axios.delete(`${API}/api/folders/history/item`, {
                data: { path: folder }
            });
            fetchFolderHistory();  // Refresh the list
        } catch (error) {
            console.error('Failed to remove folder from history:', error);
        }
    };

    const handleClearAllHistory = async () => {
        if (confirm('Clear all folder history?')) {
            try {
                await axios.delete(`${API}/api/folders/history`);
                setFolderHistory([]);
                setShowHistory(false);
            } catch (error) {
                console.error('Failed to clear folder history:', error);
            }
        }
    };

    const handleIndex = async () => {
        await handleSave();
        try {
            await axios.post(`${API}/api/index`);
        } catch (error) {
            console.error('Failed to index:', error);
        }
    };

    const handleDeleteAllHistory = async () => {
        if (confirm('Delete all search history?')) {
            try {
                await axios.delete(`${API}/api/search/history`);
                alert('History cleared!');
            } catch (error) {
                console.error('Failed to clear history:', error);
            }
        }
    };

    const handleClearCache = async () => {
        if (confirm('Clear all AI response cache? This will reset the learning.')) {
            try {
                await axios.post(`${API}/api/cache/clear`);
                await fetchCacheStats();
                alert('Cache cleared!');
            } catch (error) {
                console.error('Failed to clear cache:', error);
            }
        }
    };

    const apiProviders = [
        { id: 'openai', name: 'OpenAI (ChatGPT)', key: 'openai_api_key', placeholder: 'sk-...' },
        { id: 'gemini', name: 'Google Gemini', key: 'gemini_api_key', placeholder: 'AIza...' },
        { id: 'anthropic', name: 'Anthropic (Claude)', key: 'anthropic_api_key', placeholder: 'sk-ant-...' },
        { id: 'grok', name: 'xAI (Grok)', key: 'grok_api_key', placeholder: 'xai-...' },
    ];

    if (!isOpen) return null;

    const settingsSections = [
        { id: 'folders', label: 'Indexed Folders', icon: FolderOpen },
        { id: 'providers', label: 'AI Provider', icon: Key },
        { id: 'embeddings', label: 'Embedding Provider', icon: Database },
        { id: 'local', label: 'Local Models', icon: Cpu },
        { id: 'data', label: 'Data Management', icon: Trash2 },
    ];

    return (
        <>
            <div
                className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
                onClick={(e) => {
                    if (e.target === e.currentTarget) onClose();
                }}
                onKeyDown={(e) => {
                    if (e.key === 'Escape' && e.target === e.currentTarget) onClose();
                }}
                tabIndex={-1}
            >
                <div className="glass-overlay w-[96vw] max-w-6xl h-[88vh] rounded-2xl shadow-2xl overflow-hidden flex flex-col">
                    <div className="sticky top-0 z-20 flex items-center justify-between p-4 border-b border-border/30 bg-background/85 backdrop-blur-md">
                        <h2 className="text-lg font-bold flex items-center gap-3">
                            <div className="p-2 rounded-xl bg-primary/10">
                                <Settings className="w-5 h-5 text-primary" />
                            </div>
                            Settings
                        </h2>
                        <button type="button"
                            onClick={onClose}
                            className="p-2 rounded-xl hover:bg-secondary transition-colors"
                            aria-label="Close settings"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    </div>

                    <div className="flex-1 overflow-hidden grid grid-cols-1 md:grid-cols-[260px_1fr]">
                        <aside className="border-r border-border/30 bg-card/30 p-3 overflow-y-auto">
                            <nav className="space-y-1">
                                {settingsSections.map((section) => {
                                    const Icon = section.icon;
                                    const isActive = activeSection === section.id;
                                    return (
                                        <button type="button"
                                            key={section.id}
                                            onClick={() => setActiveSection(section.id)}
                                            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left text-sm transition-colors ${
                                                isActive
                                                    ? 'bg-primary/15 text-primary font-semibold border border-primary/20'
                                                    : 'text-muted-foreground hover:bg-secondary/60 hover:text-foreground'
                                            }`}
                                        >
                                            <Icon className="w-4 h-4" />
                                            {section.label}
                                        </button>
                                    );
                                })}
                            </nav>
                        </aside>

                        <main className="overflow-y-auto p-5 space-y-5">
                            {indexingStatus.running && (
                                <div className="p-4 rounded-xl glass-v2 border border-primary/20 bg-primary/5">
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-sm font-bold text-primary flex items-center gap-2">
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Indexing...
                                        </span>
                                        <span className="text-xs font-mono font-bold text-primary">
                                            {indexingStatus.progress}%
                                        </span>
                                    </div>
                                    <div className="w-full bg-primary/10 rounded-full h-2 overflow-hidden">
                                        <div
                                            className="bg-primary h-full transition-all duration-300"
                                            style={{ width: `${indexingStatus.progress}%` }}
                                        />
                                    </div>
                                    <div className="text-[10px] text-muted-foreground truncate opacity-80 font-mono mt-1">
                                        {indexingStatus.current_file}
                                    </div>
                                </div>
                            )}

                            {indexingStatus.error && !indexingStatus.running && (
                                <div className="p-4 rounded-xl glass-v2 border border-destructive/20 bg-destructive/5 flex items-center gap-3">
                                    <div className="p-2 rounded-full bg-destructive/10 text-destructive">
                                        <X className="w-4 h-4" />
                                    </div>
                                    <div className="text-sm font-medium text-destructive">{indexingStatus.error}</div>
                                </div>
                            )}

                            {activeSection === 'folders' && (
                                <section className="space-y-4">
                                    <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Indexed Folders</h3>
                                    <div className="flex items-center gap-2">
                                        <button type="button"
                                            onClick={handleAddFolder}
                                            className="px-3 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:opacity-90"
                                        >
                                            Add Folder
                                        </button>
                                        <button type="button"
                                            title="Previously indexed folders"
                                            onClick={() => setShowHistory((prev) => !prev)}
                                            className="px-3 py-2 rounded-lg border border-border text-sm hover:bg-secondary"
                                        >
                                            Recent History
                                        </button>
                                    </div>

                                    {showHistory && (
                                        <div className="p-3 rounded-lg border border-border bg-card/50 space-y-2">
                                            <div className="flex items-center justify-between">
                                                <h4 className="text-sm font-medium">Previously Indexed</h4>
                                                <button type="button"
                                                    onClick={handleClearAllHistory}
                                                    className="text-xs text-destructive hover:underline"
                                                >
                                                    Clear All
                                                </button>
                                            </div>
                                            {folderHistory.length === 0 ? (
                                                <p className="text-xs text-muted-foreground">No indexed folders yet.</p>
                                            ) : (
                                                folderHistory.map((folder) => (
                                                    <button type="button"
                                                        key={folder}
                                                        onClick={() => handleAddFromHistory(folder)}
                                                        className="w-full flex items-center justify-between gap-3 px-2 py-1.5 rounded-md hover:bg-secondary text-left"
                                                    >
                                                        <span className="text-xs truncate">{folder}</span>
                                                        <Trash2
                                                            className="w-3.5 h-3.5 text-destructive flex-shrink-0"
                                                            onClick={(e) => handleRemoveFromHistory(folder, e)}
                                                        />
                                                    </button>
                                                ))
                                            )}
                                        </div>
                                    )}

                                    <div className="space-y-2">
                                        {config.folders.length === 0 ? (
                                            <p className="text-sm text-muted-foreground">No folders selected. Add a folder to build your index.</p>
                                        ) : (
                                            config.folders.map((folder) => (
                                                <div key={folder} className="flex items-center justify-between gap-3 p-3 rounded-lg border border-border bg-card/30">
                                                    <div className="flex items-center gap-2 min-w-0">
                                                        <FolderOpen className="w-4 h-4 text-primary flex-shrink-0" />
                                                        <span className="text-sm truncate">{folder}</span>
                                                    </div>
                                                    <button type="button"
                                                        onClick={() => handleRemoveFolder(folder)}
                                                        aria-label={`Remove ${folder} from index`}
                                                        className="p-1.5 rounded-md hover:bg-destructive/10 text-destructive"
                                                    >
                                                        <X className="w-4 h-4" />
                                                    </button>
                                                </div>
                                            ))
                                        )}
                                    </div>

                                    <div className="pt-2">
                                        <button type="button"
                                            onClick={handleIndex}
                                            disabled={isLoading || indexingStatus.running}
                                            className="px-4 py-2 rounded-lg btn-cosmic text-sm font-medium disabled:opacity-60"
                                        >
                                            Rebuild Index
                                        </button>
                                    </div>
                                </section>
                            )}

                            {activeSection === 'providers' && (
                                <section className="space-y-4">
                                    <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">API Keys</h3>
                                    {apiProviders.map(provider => (
                                        <div key={provider.id} className="space-y-1">
                                            <label htmlFor={provider.id} className="text-sm font-medium">{provider.name}</label>
                                            <input
                                                id={provider.id}
                                                type="password"
                                                value={config[provider.key] || ''}
                                                onChange={(e) => setConfig({ ...config, [provider.key]: e.target.value })}
                                                className="w-full px-3 py-2 text-sm rounded-lg border border-input bg-background"
                                                placeholder={provider.placeholder}
                                            />
                                        </div>
                                    ))}
                                    <p className="text-xs text-muted-foreground">API keys are stored locally and used only when that provider is selected.</p>
                                </section>
                            )}

                            {activeSection === 'embeddings' && (
                                <section className="space-y-4" id="embedding-config-panel">
                                    <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Embedding Provider</h3>
                                    <div className="space-y-1">
                                        <label className="text-xs font-medium text-muted-foreground" htmlFor="embedding-provider-select">Provider Type</label>
                                        <select
                                            id="embedding-provider-select"
                                            value={embeddingConfig.provider_type}
                                            onChange={(e) => setEmbeddingConfig(prev => ({ ...prev, provider_type: e.target.value, api_key: '' }))}
                                            className="w-full px-3 py-2 text-sm rounded-lg border border-input bg-background"
                                        >
                                            {EMBEDDING_PROVIDER_TYPES.map(pt => (
                                                <option key={pt.value} value={pt.value}>{pt.label}</option>
                                            ))}
                                        </select>
                                    </div>

                                    <div className="space-y-1">
                                        <label className="text-xs font-medium text-muted-foreground" htmlFor="embedding-model-name">Model Name / Repo ID</label>
                                        <input
                                            id="embedding-model-name"
                                            type="text"
                                            value={embeddingConfig.model_name}
                                            onChange={(e) => setEmbeddingConfig(prev => ({ ...prev, model_name: e.target.value }))}
                                            className="w-full px-3 py-2 text-sm rounded-lg border border-input bg-background font-mono"
                                            placeholder="e.g. Alibaba-NLP/gte-Qwen2-1.5B-instruct"
                                        />
                                    </div>

                                    {EMBEDDING_PROVIDER_TYPES.find(pt => pt.value === embeddingConfig.provider_type)?.needsKey && (
                                        <div className="space-y-1">
                                            <label className="text-xs font-medium text-muted-foreground" htmlFor="embedding-api-key">
                                                API Key
                                                {embeddingConfig.api_key === '••••••••' && (
                                                    <span className="ml-2 text-[10px] bg-green-500/10 text-green-600 px-1.5 py-0.5 rounded font-semibold">Stored</span>
                                                )}
                                            </label>
                                            <input
                                                id="embedding-api-key"
                                                type="password"
                                                value={embeddingConfig.api_key}
                                                onChange={(e) => setEmbeddingConfig(prev => ({ ...prev, api_key: e.target.value }))}
                                                className="w-full px-3 py-2 text-sm rounded-lg border border-input bg-background"
                                                placeholder={embeddingConfig.api_key === '••••••••' ? 'Leave blank to keep current key' : 'Paste your API key'}
                                            />
                                        </div>
                                    )}

                                    <p className="text-[11px] text-muted-foreground leading-relaxed">The embedding provider is used to convert documents and queries into vectors for semantic search. Changes take effect on the next index rebuild.</p>
                                </section>
                            )}

                            {activeSection === 'local' && (
                                <section className="space-y-4">
                                    <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Local Model Settings</h3>
                                    {activeModel && (
                                        <div className="p-4 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-between">
                                            <div>
                                                <div className="text-[10px] uppercase font-bold text-primary tracking-wider mb-0.5">Currently Active</div>
                                                <div className="text-sm font-bold flex items-center gap-2">
                                                    <Cpu className="w-4 h-4 text-primary" />
                                                    {activeModel}
                                                </div>
                                            </div>
                                            <div className="px-2 py-0.5 bg-primary text-primary-foreground text-[10px] rounded-md font-bold uppercase tracking-widest">Active</div>
                                        </div>
                                    )}
                                    <ModelManager
                                        onSelectModel={(path) => setConfig({ ...config, local_model_path: path })}
                                        activeModel={activeModel}
                                        selectedPath={config.local_model_path}
                                    />
                                </section>
                            )}

                            {activeSection === 'data' && (
                                <section className="space-y-4">
                                    <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Data Management</h3>
                                    <div className="flex items-center justify-between p-3 rounded-lg border border-border bg-card/50">
                                        <div>
                                            <div className="text-sm font-semibold flex items-center gap-2"><span className="text-primary">⚡</span> AI Response Cache</div>
                                            <div className="text-xs text-muted-foreground">{cacheStats.total_entries} entries • {cacheStats.total_hits} hits saved</div>
                                        </div>
                                        <button type="button"
                                            onClick={handleClearCache}
                                            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-destructive/10 text-destructive hover:bg-destructive/20 transition-colors"
                                        >
                                            Clear Cache
                                        </button>
                                    </div>
                                    <button type="button"
                                        onClick={handleDeleteAllHistory}
                                        className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-destructive hover:bg-destructive/10 transition-colors"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                        Clear Search History
                                    </button>
                                </section>
                            )}
                        </main>
                    </div>

                    <div className="sticky bottom-0 p-4 border-t border-border/30 bg-background/85 backdrop-blur-md flex justify-end gap-3">
                        <button type="button"
                            onClick={onClose}
                            className="px-4 py-2.5 rounded-xl text-sm font-medium hover:bg-secondary transition-colors"
                        >
                            Cancel
                        </button>
                        <button type="button"
                            onClick={handleSave}
                            disabled={isLoading}
                            className="px-5 py-2.5 rounded-xl btn-cosmic text-sm font-semibold flex items-center gap-2 disabled:opacity-50"
                        >
                            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                            Save Changes
                        </button>
                    </div>
                </div>
            </div>

            {toast && (
                <Toast
                    message={toast.message}
                    type={toast.type}
                    onDismiss={dismissToast}
                />
            )}
        </>
    );
};

export default SettingsModal;
