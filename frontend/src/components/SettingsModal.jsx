import React, { useEffect, useState } from 'react';
import { X, Folder, Link2, Check, Settings, Trash2, Shield, Cloud, Cpu, Sparkles, Layers, History, RefreshCw, AlertCircle, Sun, Moon, Monitor, Palette } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';
import ModelManager from './ModelManager';

const PROVIDERS = [
    { id: 'openai', label: 'OpenAI', icon: Cloud, keyField: 'openai_api_key', placeholder: 'sk-...' },
    { id: 'gemini', label: 'Google Gemini', icon: Sparkles, keyField: 'gemini_api_key', placeholder: 'AIza...' },
    { id: 'anthropic', label: 'Anthropic Claude', icon: Cloud, keyField: 'anthropic_api_key', placeholder: 'sk-ant-...' },
    { id: 'grok', label: 'xAI Grok', icon: Cloud, keyField: 'grok_api_key', placeholder: 'xai-...' },
];

const DEFAULT_EMBEDDING_PRESETS = [
    {
        id: 'fast',
        model_name: 'sentence-transformers/all-MiniLM-L6-v2',
        label: 'Fast (MiniLM-L6)',
        ram_tier: 'Compatible with 8GB',
        dim: 384,
        context_tokens: 256,
        size: '~90 MB',
        speed: '⚡ Ultra Fast',
        retrieval: 'Good',
        description: 'Blazing speed on CPU. Great default for simple documents and quick indexing.',
    },
    {
        id: 'balanced',
        model_name: 'BAAI/bge-small-en-v1.5',
        label: 'Balanced (BGE-Small v1.5)',
        ram_tier: 'Compatible with 8GB',
        dim: 384,
        context_tokens: 512,
        size: '~130 MB',
        speed: '⚡ Ultra Fast',
        retrieval: 'High',
        description: 'Stronger semantics than MiniLM with 2x context window, while remaining ultra fast on CPU.',
    },
    {
        id: 'nomic',
        model_name: 'nomic-ai/nomic-embed-text-v1.5',
        label: 'SOTA RAG & Long Docs (Nomic Embed v1.5)',
        ram_tier: 'Compatible with 16GB',
        dim: 768,
        context_tokens: 8192,
        size: '~550 MB',
        speed: '⚡ Fast',
        retrieval: 'Very High',
        recommended: true,
        description: 'Premier open-source model for document RAG. Massive 8k token context window; outperforms OpenAI text-embedding-3-small.',
    },
    {
        id: 'quality',
        model_name: 'BAAI/bge-base-en-v1.5',
        label: 'Quality (BGE-Base v1.5)',
        ram_tier: 'Compatible with 16GB',
        dim: 768,
        context_tokens: 512,
        size: '~440 MB',
        speed: '⚡ Fast',
        retrieval: 'Very High',
        description: 'High-accuracy dense semantic retrieval. Ideal for technical research and detailed contracts.',
    },
    {
        id: 'bge-m3',
        model_name: 'BAAI/bge-m3',
        label: 'Elite Multilingual (BGE-M3)',
        ram_tier: 'Compatible with 32GB+',
        dim: 1024,
        context_tokens: 8192,
        size: '~2.2 GB',
        speed: '⚡ Moderate',
        retrieval: 'Elite',
        description: 'Multi-functionality (dense, lexical BM25, multi-vector) across 100+ languages with 8k token context.',
    },
    {
        id: 'max',
        model_name: 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
        label: 'Max (GTE-Qwen2 1.5B)',
        ram_tier: 'Compatible with 32GB+',
        dim: 1536,
        context_tokens: 32768,
        size: '~6 GB',
        speed: '⚡ Moderate',
        retrieval: 'Elite',
        description: 'Leaderboard-topping retrieval precision. Best when GPU acceleration is available.',
    },
];

const EMBEDDING_RAM_TIERS = [
    { id: 'all', label: 'All Models' },
    { id: '8gb', label: 'Compatible with 8GB' },
    { id: '16gb', label: 'Compatible with 16GB' },
    { id: '32gb', label: 'Compatible with 32GB+' },
];

export default function SettingsModal({ isOpen, onClose, onSaved, darkMode, setDarkMode, onToggleDark }) {
    const [config, setConfig] = useState(null);
    const [embeddingConfig, setEmbeddingConfig] = useState(null);
    const [initialEmbeddingModel, setInitialEmbeddingModel] = useState('');
    const [embeddingPresets, setEmbeddingPresets] = useState(DEFAULT_EMBEDDING_PRESETS);
    const [embeddingTier, setEmbeddingTier] = useState('all');
    const [reindexing, setReindexing] = useState(false);
    const [themePreference, setThemePreference] = useState(() => {
        return localStorage.getItem('docu-ai-theme') || 'system';
    });
    const [cacheStats, setCacheStats] = useState({ total_entries: 0, total_hits: 0 });
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [activeTab, setActiveTab] = useState('folders');
    const [newFolder, setNewFolder] = useState('');
    const [folderHistory, setFolderHistory] = useState([]);
    const [showFolderHistory, setShowFolderHistory] = useState(false);
    const [validatingFolder, setValidatingFolder] = useState(false);
    const toast = useToast();

    const handleThemeChange = (mode) => {
        setThemePreference(mode);
        if (mode === 'system') {
            localStorage.removeItem('docu-ai-theme');
            const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            setDarkMode?.(systemDark);
            if (systemDark) document.documentElement.classList.add('dark');
            else document.documentElement.classList.remove('dark');
        } else if (mode === 'dark') {
            localStorage.setItem('docu-ai-theme', 'dark');
            setDarkMode?.(true);
            document.documentElement.classList.add('dark');
        } else {
            localStorage.setItem('docu-ai-theme', 'light');
            setDarkMode?.(false);
            document.documentElement.classList.remove('dark');
        }
    };

    useEffect(() => {
        if (!isOpen) return;
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', handleKeyDown);
        load();
        return () => window.removeEventListener('keydown', handleKeyDown);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    const load = async () => {
        setLoading(true);
        try {
            const [cRes, eRes, hRes, cacheRes, pRes] = await Promise.all([
                api.getConfig(),
                api.getEmbeddingConfig(),
                api.getFolderHistory(),
                api.getCacheStats(),
                api.getEmbeddingPresets().catch(() => ({ data: { presets: [] } }))
            ]);
            setConfig(cRes.data);
            setEmbeddingConfig(eRes.data);
            setInitialEmbeddingModel(eRes.data?.model_name || '');
            setFolderHistory(hRes.data || []);
            setCacheStats(cacheRes.data);
            if (pRes?.data?.presets && pRes.data.presets.length > 0) {
                setEmbeddingPresets(pRes.data.presets);
            }
        } catch {
            toast.error('Could not load settings');
        } finally {
            setLoading(false);
        }
    };

    const save = async () => {
        setSaving(true);
        let configSaved = false;
        try {
            await api.saveConfig(config);
            configSaved = true;
            await api.saveEmbeddingConfig(embeddingConfig);
            if (embeddingConfig?.model_name && embeddingConfig.model_name !== initialEmbeddingModel) {
                setInitialEmbeddingModel(embeddingConfig.model_name);
                toast.info('Embedding model updated! Re-indexing is recommended.');
            }
            window.dispatchEvent(new CustomEvent('docu-config-updated'));
            toast.success('Settings saved');
            onSaved?.();
        } catch (e) {
            toast.error(
                e.response?.data?.detail ||
                (configSaved
                    ? 'General settings saved, but embedding settings failed to save'
                    : 'Could not save settings')
            );
        } finally {
            setSaving(false);
        }
    };

    const handleReindex = async () => {
        setReindexing(true);
        try {
            if (embeddingConfig) {
                await api.saveEmbeddingConfig(embeddingConfig);
                setInitialEmbeddingModel(embeddingConfig.model_name);
            }
            await api.startIndexing();
            window.dispatchEvent(new CustomEvent('docu-config-updated'));
            toast.success('Re-indexing started with selected embedding model');
        } catch {
            toast.error('Could not start re-indexing');
        } finally {
            setReindexing(false);
        }
    };

    const addFolder = async (path = newFolder) => {
        if (!path) return;
        setValidatingFolder(true);
        try {
            const res = await api.validatePath(path);
            if (res.data && res.data.valid === false) {
                toast.error(res.data.error || 'Invalid folder path');
                return;
            }
            setConfig((c) => ({
                ...c,
                folders: [...new Set([...(c.folders || []), path])],
            }));
            setNewFolder('');
            setShowFolderHistory(false);
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Invalid folder path');
        } finally {
            setValidatingFolder(false);
        }
    };

    const handleBrowse = async () => {
        try {
            const res = await api.browse();
            if (res.data && res.data.folder) {
                // If they picked a folder, add it automatically
                addFolder(res.data.folder);
            }
        } catch (e) {
            toast.error('Could not open folder browser');
        }
    };

    const removeFolder = (path) => {
        setConfig((c) => ({ ...c, folders: c.folders.filter((x) => x !== path) }));
    };

    const clearFolderHistory = async () => {
        if (!confirm('Clear all previously indexed folders history?')) return;
        try {
            await api.clearFolderHistory();
            setFolderHistory([]);
            toast.success('Folder history cleared');
        } catch {
            toast.error('Could not clear history');
        }
    };

    const clearCache = async () => {
        try {
            await api.clearCache();
            const cacheRes = await api.getCacheStats();
            setCacheStats(cacheRes.data);
            toast.success('Cache cleared');
        } catch {
            toast.error('Could not clear cache');
        }
    };

    const updateProviderKey = (keyField, value) => {
        setConfig((c) => ({ ...c, [keyField]: value }));
    };

    const handleDeleteKey = async (provider) => {
        if (!confirm(`Delete ${provider.label} API key?`)) return;
        try {
            const res = await api.deleteApiKey(provider.id);
            setConfig((c) => ({
                ...c,
                [provider.keyField]: '',
                [`${provider.keyField}_set`]: false,
                provider: res.data?.provider || (c.provider === provider.id ? 'local' : c.provider),
            }));
            window.dispatchEvent(new CustomEvent('docu-config-updated'));
            toast.success(`${provider.label} API key deleted`);
            onSaved?.();
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not delete API key');
        }
    };

    if (!isOpen) return null;

    return (
        <>
            <div className="fixed inset-0 bg-[rgba(0,0,0,0.4)] dark:bg-[rgba(0,0,0,0.6)] z-50 backdrop-blur-sm transition-opacity" onClick={onClose} />
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 pointer-events-none" role="dialog" aria-label="Settings">
                <div className="bg-canvas dark:bg-[#111111] w-full max-w-4xl max-h-[90vh] rounded-v-lg shadow-v-5 dark:shadow-v-dark-5 flex flex-col pointer-events-auto border border-hairline dark:border-[rgba(255,255,255,0.1)] overflow-hidden animate-slide-up">
                    
                    {/* Header */}
                    <div className="flex items-center justify-between px-6 py-4 border-b border-hairline dark:border-[rgba(255,255,255,0.08)]">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-v-sm bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.06)] flex items-center justify-center border border-hairline dark:border-[rgba(255,255,255,0.1)]">
                                <Settings className="w-4 h-4 text-ink dark:text-[#ededed]" />
                            </div>
                            <h2 className="font-semibold text-lg text-ink dark:text-[#ededed] tracking-[-0.3px]">Settings</h2>
                        </div>
                        <button onClick={onClose} aria-label="Close settings" className="p-2 rounded-v-sm text-mute hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)] hover:text-ink dark:hover:text-[#ededed] transition">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {loading || !config || !embeddingConfig ? (
                        <div className="p-12 text-center text-mute flex flex-col items-center justify-center flex-1">
                            <div className="w-8 h-8 rounded-full border-2 border-mute border-t-transparent animate-spin mb-4" />
                            Loading configuration...
                        </div>
                    ) : (
                        <div className="flex flex-1 overflow-hidden flex-col sm:flex-row">
                            {/* Sidebar tabs */}
                            <div className="w-full sm:w-56 bg-canvas-soft dark:bg-[#0a0a0a] border-b sm:border-b-0 sm:border-r border-hairline dark:border-[rgba(255,255,255,0.08)] flex-shrink-0 p-3 flex sm:flex-col gap-1 overflow-x-auto sm:overflow-y-auto no-scrollbar">
                                {[
                                    { id: 'folders', icon: Folder, label: 'Folders' },
                                    { id: 'embeddings', icon: Layers, label: 'Embeddings' },
                                    { id: 'providers', icon: Shield, label: 'Cloud Providers' },
                                    { id: 'external', icon: Link2, label: 'External (Ollama)' },
                                    { id: 'local', icon: Cpu, label: 'Local Models' },
                                    { id: 'appearance', icon: Palette, label: 'Appearance' },
                                    { id: 'system', icon: Settings, label: 'System' },
                                ].map((t) => {
                                    const Icon = t.icon;
                                    const active = activeTab === t.id;
                                    return (
                                        <button
                                            key={t.id}
                                            onClick={() => setActiveTab(t.id)}
                                            className={`flex items-center gap-2.5 px-3 py-2 rounded-v-sm text-[13px] font-medium transition whitespace-nowrap ${
                                                active
                                                    ? 'bg-canvas dark:bg-[#1a1a1a] text-ink dark:text-[#ededed] shadow-v-1 dark:shadow-[inset_0_0_0_1px_rgba(255,255,255,0.1)]'
                                                    : 'text-body dark:text-[#888] hover:bg-[rgba(0,0,0,0.04)] dark:hover:bg-[rgba(255,255,255,0.04)] hover:text-ink dark:hover:text-[#ededed]'
                                            }`}
                                        >
                                            <Icon className="w-4 h-4 flex-shrink-0" />
                                            {t.label}
                                        </button>
                                    );
                                })}
                            </div>

                            {/* Content area */}
                            <div className="flex-1 overflow-y-auto p-6 bg-canvas dark:bg-[#111111]">
                                <div className="max-w-2xl">
                                    
                                    {activeTab === 'folders' && (
                                        <div className="space-y-6">
                                            <div>
                                                <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">Indexed Folders</h3>
                                                <p className="text-sm text-body dark:text-[#888] mb-4">Directories the search engine will scan and index.</p>
                                                
                                                <div className="flex gap-2 mb-4 relative">
                                                    <input
                                                        type="text"
                                                        className="input flex-1"
                                                        placeholder="e.g. C:\Users\siddh\Documents"
                                                        value={newFolder}
                                                        onChange={(e) => setNewFolder(e.target.value)}
                                                        onKeyDown={(e) => e.key === 'Enter' && addFolder()}
                                                    />
                                                    <button onClick={handleBrowse} type="button" className="btn-ghost whitespace-nowrap">
                                                        Browse
                                                    </button>
                                                    <button onClick={() => addFolder(newFolder)} disabled={validatingFolder} className="btn-secondary whitespace-nowrap disabled:opacity-50">
                                                        {validatingFolder ? 'Validating…' : 'Add folder'}
                                                    </button>
                                                    {folderHistory.length > 0 && (
                                                        <button 
                                                            title="Previously indexed folders"
                                                            className="btn-ghost px-2"
                                                            onClick={() => setShowFolderHistory(!showFolderHistory)}
                                                        >
                                                            <History className="w-4 h-4" />
                                                        </button>
                                                    )}
                                                    
                                                    {showFolderHistory && (
                                                        <div className="absolute right-0 top-full mt-1 z-10 w-72 bg-canvas dark:bg-[#171717] border border-hairline dark:border-[rgba(255,255,255,0.1)] rounded-v-md shadow-v-4 p-2 animate-slide-up">
                                                            <div className="flex items-center justify-between px-2 mb-2">
                                                                <span className="text-[10px] font-mono uppercase text-mute">History</span>
                                                                <button onClick={clearFolderHistory} className="text-[10px] font-mono uppercase text-error hover:underline">Clear All</button>
                                                            </div>
                                                            <ul className="max-h-48 overflow-y-auto space-y-1">
                                                                {folderHistory.map(h => (
                                                                    <li key={h}>
                                                                        <button
                                                                            className="w-full text-left px-2 py-1.5 text-xs text-ink dark:text-[#ededed] hover:bg-canvas-soft dark:hover:bg-[rgba(255,255,255,0.06)] rounded-v-sm truncate"
                                                                            onClick={() => addFolder(h)}
                                                                        >
                                                                            {h}
                                                                        </button>
                                                                    </li>
                                                                ))}
                                                            </ul>
                                                        </div>
                                                    )}
                                                </div>

                                                {(!config.folders || config.folders.length === 0) ? (
                                                    <div className="card p-6 text-center text-sm text-mute">No folders added.</div>
                                                ) : (
                                                    <ul className="card overflow-hidden">
                                                        {config.folders.map((f, i) => (
                                                            <li key={i} className="flex items-center justify-between p-3 border-b border-hairline dark:border-[rgba(255,255,255,0.08)] last:border-0 hover:bg-canvas-soft dark:hover:bg-[rgba(255,255,255,0.02)] transition">
                                                                <div className="font-mono text-[13px] text-ink dark:text-[#ededed] truncate mr-3" title={f}>{f}</div>
                                                                <button aria-label={`Remove ${f} from index`} onClick={() => removeFolder(f)} className="p-1.5 text-mute hover:text-error hover:bg-error-soft dark:hover:bg-[rgba(238,0,0,0.1)] rounded-v-sm transition flex-shrink-0">
                                                                    <Trash2 className="w-4 h-4" />
                                                                </button>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                )}
                                            </div>

                                            <label className="flex items-start gap-3 p-4 card cursor-pointer hover:border-hairline-strong dark:hover:border-[rgba(255,255,255,0.2)] transition">
                                                <div className="mt-0.5">
                                                    <input
                                                        type="checkbox"
                                                        className="w-4 h-4 rounded border-hairline text-ink focus:ring-ink"
                                                        checked={config.auto_index}
                                                        onChange={(e) => setConfig({ ...config, auto_index: e.target.checked })}
                                                    />
                                                </div>
                                                <div>
                                                    <div className="font-medium text-ink dark:text-[#ededed] text-sm">Auto-index on changes</div>
                                                    <div className="text-xs text-mute mt-0.5">Watch folders for real-time changes. Uses more background CPU.</div>
                                                </div>
                                            </label>
                                        </div>
                                    )}

                                    {activeTab === 'embeddings' && (() => {
                                        const activePresets = (embeddingPresets && embeddingPresets.length > 0) ? embeddingPresets : DEFAULT_EMBEDDING_PRESETS;
                                        const filteredEmbeddingPresets = activePresets.filter((preset) => {
                                            if (embeddingTier === 'all') return true;
                                            if (embeddingTier === '8gb') return preset.ram_tier?.includes('8GB') || preset.dim <= 384;
                                            if (embeddingTier === '16gb') return preset.ram_tier?.includes('16GB') || (preset.dim > 384 && preset.dim <= 768);
                                            if (embeddingTier === '32gb') return preset.ram_tier?.includes('32GB') || preset.dim > 768;
                                            return true;
                                        });
                                        const isModelChanged = Boolean(
                                            embeddingConfig?.model_name &&
                                            initialEmbeddingModel &&
                                            embeddingConfig.model_name !== initialEmbeddingModel
                                        );
                                        const getBadgeStyle = (tier) => {
                                            if (tier?.includes('8GB')) return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20';
                                            if (tier?.includes('16GB')) return 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20';
                                            return 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20';
                                        };

                                        return (
                                            <div className="space-y-6">
                                                <div>
                                                    <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">Embeddings</h3>
                                                    <p className="text-sm text-body dark:text-[#888]">
                                                        Embeddings turn documents into high-dimensional vectors for semantic search. Models are cached locally on first use and run 100% offline.
                                                    </p>
                                                </div>

                                                {/* Re-indexing notice / status banner */}
                                                {isModelChanged ? (
                                                    <div className="p-4 rounded-v-sm bg-amber-500/10 border border-amber-500/20 text-amber-900 dark:text-amber-200 text-sm space-y-3">
                                                        <div className="flex items-start gap-2.5">
                                                            <AlertCircle className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
                                                            <div>
                                                                <p className="font-medium">Embedding model changed</p>
                                                                <p className="text-xs text-amber-800/80 dark:text-amber-300/80 mt-0.5">
                                                                    You switched from <span className="font-mono">{initialEmbeddingModel}</span> to <span className="font-mono font-semibold">{embeddingConfig.model_name}</span>. Because vector dimensions and math spaces differ, your existing index must be regenerated.
                                                                </p>
                                                            </div>
                                                        </div>
                                                        <button
                                                            type="button"
                                                            onClick={handleReindex}
                                                            disabled={reindexing}
                                                            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-v-sm bg-amber-500 text-white font-medium text-xs hover:bg-amber-600 transition disabled:opacity-50"
                                                        >
                                                            <RefreshCw className={`w-3.5 h-3.5 ${reindexing ? 'animate-spin' : ''}`} />
                                                            {reindexing ? 'Starting Re-index...' : 'Save & Re-index Now'}
                                                        </button>
                                                    </div>
                                                ) : (
                                                    <div className="p-3.5 rounded-v-sm bg-canvas-soft dark:bg-[rgba(255,255,255,0.04)] border border-hairline dark:border-[rgba(255,255,255,0.1)] text-xs text-body dark:text-[#888] flex items-center justify-between gap-4">
                                                        <div className="flex items-center gap-2">
                                                            <Layers className="w-4 h-4 text-emerald-500 shrink-0" />
                                                            <span>Active Model: <strong className="text-ink dark:text-[#ededed] font-mono">{embeddingConfig.model_name || 'sentence-transformers/all-MiniLM-L6-v2'}</strong></span>
                                                        </div>
                                                        <button
                                                            type="button"
                                                            onClick={handleReindex}
                                                            disabled={reindexing}
                                                            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-v-sm border border-hairline dark:border-[rgba(255,255,255,0.1)] text-xs hover:bg-canvas dark:hover:bg-[rgba(255,255,255,0.08)] text-body dark:text-[#888] hover:text-ink dark:hover:text-[#ededed] transition"
                                                        >
                                                            <RefreshCw className={`w-3 h-3 ${reindexing ? 'animate-spin' : ''}`} />
                                                            {reindexing ? 'Re-indexing...' : 'Re-index Documents'}
                                                        </button>
                                                    </div>
                                                )}

                                                {/* Provider Mode Selection */}
                                                <div className="card p-4 space-y-4">
                                                    <label className="label">Provider Mode</label>
                                                    <div className="flex gap-2">
                                                        <button
                                                            onClick={() => setEmbeddingConfig({ ...embeddingConfig, provider_type: 'local' })}
                                                            className={`flex-1 py-2 rounded-v-sm text-sm font-medium transition ${embeddingConfig.provider_type === 'local' ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink' : 'bg-canvas-soft dark:bg-[rgba(255,255,255,0.04)] border border-hairline dark:border-[rgba(255,255,255,0.1)] text-body dark:text-[#888] hover:bg-canvas dark:hover:bg-[rgba(255,255,255,0.08)]'}`}
                                                        >
                                                            Local (on-device)
                                                        </button>
                                                        <button
                                                            onClick={() => setEmbeddingConfig({ ...embeddingConfig, provider_type: 'huggingface_api' })}
                                                            className={`flex-1 py-2 rounded-v-sm text-sm font-medium transition ${embeddingConfig.provider_type === 'huggingface_api' ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink' : 'bg-canvas-soft dark:bg-[rgba(255,255,255,0.04)] border border-hairline dark:border-[rgba(255,255,255,0.1)] text-body dark:text-[#888] hover:bg-canvas dark:hover:bg-[rgba(255,255,255,0.08)]'}`}
                                                        >
                                                            HuggingFace API
                                                        </button>
                                                    </div>

                                                    {embeddingConfig.provider_type !== 'local' && (
                                                        <div>
                                                            <label htmlFor="api_key" className="label">API Key</label>
                                                            <input
                                                                id="api_key"
                                                                type="password"
                                                                className="input font-mono text-sm"
                                                                value={embeddingConfig.api_key || ''}
                                                                onChange={(e) => setEmbeddingConfig({ ...embeddingConfig, api_key: e.target.value })}
                                                                placeholder="hf_..."
                                                            />
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Curated Presets & RAM Tier Navigation */}
                                                <div className="space-y-4">
                                                    <div className="flex items-center justify-between gap-2 flex-wrap">
                                                        <div>
                                                            <h4 className="text-sm font-semibold text-ink dark:text-[#ededed]">Curated Open-Source Models</h4>
                                                            <p className="text-xs text-body dark:text-[#888]">Filter by system RAM compatibility</p>
                                                        </div>
                                                        <div className="flex flex-wrap gap-1.5">
                                                            {EMBEDDING_RAM_TIERS.map((tier) => (
                                                                <button
                                                                    key={tier.id}
                                                                    type="button"
                                                                    onClick={() => setEmbeddingTier(tier.id)}
                                                                    className={`px-2.5 py-1 text-xs rounded-v-sm font-medium transition-all ${
                                                                        embeddingTier === tier.id
                                                                            ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink shadow-sm'
                                                                            : 'bg-canvas-soft dark:bg-[rgba(255,255,255,0.04)] border border-hairline dark:border-[rgba(255,255,255,0.1)] text-body dark:text-[#888] hover:bg-canvas dark:hover:bg-[rgba(255,255,255,0.08)] hover:text-ink dark:hover:text-[#ededed]'
                                                                    }`}
                                                                >
                                                                    {tier.label}
                                                                </button>
                                                            ))}
                                                        </div>
                                                    </div>

                                                    {/* Preset Cards Grid */}
                                                    <div className="grid grid-cols-1 gap-3">
                                                        {filteredEmbeddingPresets.map((preset) => {
                                                            const isSelected = embeddingConfig.model_name === preset.model_name;
                                                            return (
                                                                <div
                                                                    key={preset.id}
                                                                    onClick={() => setEmbeddingConfig({ ...embeddingConfig, model_name: preset.model_name })}
                                                                    className={`p-4 rounded-v-sm border cursor-pointer transition-all ${
                                                                        isSelected
                                                                            ? 'border-emerald-500 bg-emerald-500/5 dark:bg-emerald-500/10 shadow-sm'
                                                                            : 'border-hairline dark:border-[rgba(255,255,255,0.1)] bg-canvas-soft dark:bg-[rgba(255,255,255,0.03)] hover:border-ink/30 dark:hover:border-white/30'
                                                                    }`}
                                                                >
                                                                    <div className="flex items-start justify-between gap-3 mb-2">
                                                                        <div className="space-y-1">
                                                                            <div className="flex items-center gap-2 flex-wrap">
                                                                                <h5 className="text-sm font-semibold text-ink dark:text-[#ededed]">
                                                                                    {preset.label}
                                                                                </h5>
                                                                                {preset.recommended && (
                                                                                    <span className="px-1.5 py-0.5 text-[10px] font-medium rounded-v-sm bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                                                                                        ⭐ Recommended for RAG
                                                                                    </span>
                                                                                )}
                                                                                {isSelected && (
                                                                                    <span className="px-1.5 py-0.5 text-[10px] font-medium rounded-v-sm bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                                                                                        <Check className="w-3 h-3" /> Selected
                                                                                    </span>
                                                                                )}
                                                                            </div>
                                                                            <p className="font-mono text-xs text-body dark:text-[#888] truncate max-w-md">
                                                                                {preset.model_name}
                                                                            </p>
                                                                        </div>
                                                                        <span className={`px-2 py-0.5 text-[11px] font-medium rounded-v-sm border shrink-0 ${getBadgeStyle(preset.ram_tier)}`}>
                                                                            {preset.ram_tier}
                                                                        </span>
                                                                    </div>

                                                                    <p className="text-xs text-body dark:text-[#888] mb-3 leading-relaxed">
                                                                        {preset.description}
                                                                    </p>

                                                                    <div className="flex items-center gap-2 flex-wrap text-[11px] text-body dark:text-[#888]">
                                                                        <span className="px-2 py-0.5 rounded-v-sm bg-canvas dark:bg-[rgba(255,255,255,0.06)] border border-hairline dark:border-[rgba(255,255,255,0.1)] font-mono">
                                                                            {preset.dim}-dim
                                                                        </span>
                                                                        <span className="px-2 py-0.5 rounded-v-sm bg-canvas dark:bg-[rgba(255,255,255,0.06)] border border-hairline dark:border-[rgba(255,255,255,0.1)]">
                                                                            Context: {preset.context_tokens ? `${preset.context_tokens.toLocaleString()} tokens` : `${preset.dim} tokens`}
                                                                        </span>
                                                                        <span className="px-2 py-0.5 rounded-v-sm bg-canvas dark:bg-[rgba(255,255,255,0.06)] border border-hairline dark:border-[rgba(255,255,255,0.1)]">
                                                                            {preset.size}
                                                                        </span>
                                                                        {preset.speed && (
                                                                            <span className="px-2 py-0.5 rounded-v-sm bg-canvas dark:bg-[rgba(255,255,255,0.06)] border border-hairline dark:border-[rgba(255,255,255,0.1)] font-medium text-ink dark:text-[#ededed]">
                                                                                {preset.speed}
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                </div>
                                                            );
                                                        })}
                                                    </div>

                                                    {/* Custom Model Input */}
                                                    <div className="card p-4 space-y-2 mt-4">
                                                        <label htmlFor="model_name" className="label">Model name</label>
                                                        <input
                                                            id="model_name"
                                                            type="text"
                                                            className="input font-mono text-sm"
                                                            value={embeddingConfig.model_name || ''}
                                                            onChange={(e) => setEmbeddingConfig({ ...embeddingConfig, model_name: e.target.value })}
                                                            placeholder="e.g. nomic-ai/nomic-embed-text-v1.5 or BAAI/bge-small-en-v1.5"
                                                        />
                                                        <p className="text-xs text-body dark:text-[#888]">
                                                            Selecting a preset updates the model name above. You can also specify any HuggingFace model repository.
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })()}

                                    {activeTab === 'providers' && (
                                        <div className="space-y-6">
                                            <div>
                                                <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">Cloud Providers</h3>
                                                <p className="text-sm text-body dark:text-[#888] mb-4">Enter API keys to enable cloud-based LLM generation. Keys are stored locally.</p>
                                            </div>
                                            
                                            <div className="space-y-4">
                                                {PROVIDERS.map(p => {
                                                    const isKeySet = Boolean(config[`${p.keyField}_set`] || config[p.keyField]);
                                                    const isActive = config.provider === p.id;
                                                    return (
                                                        <div key={p.id} className="card p-4">
                                                            <div className="flex items-center justify-between mb-3">
                                                                <label className="label flex items-center gap-2 mb-0 text-ink dark:text-[#ededed] font-sans font-medium tracking-normal text-sm normal-case">
                                                                    <p.icon className="w-4 h-4" />
                                                                    {p.label}
                                                                </label>
                                                                <div className="flex items-center gap-2">
                                                                    {isActive ? (
                                                                        <span className="text-[11px] font-medium px-2 py-0.5 rounded bg-ink dark:bg-[#ededed] text-white dark:text-ink">
                                                                            Active Model
                                                                        </span>
                                                                    ) : (
                                                                        <button
                                                                            type="button"
                                                                            onClick={() => setConfig((c) => ({ ...c, provider: p.id }))}
                                                                            disabled={!isKeySet}
                                                                            className="text-xs text-mute hover:text-ink dark:hover:text-[#ededed] disabled:opacity-40 disabled:hover:text-mute underline transition"
                                                                            title={isKeySet ? `Switch active search model to ${p.label}` : 'Enter an API key first to activate'}
                                                                        >
                                                                            Set as active
                                                                        </button>
                                                                    )}
                                                                    {isKeySet && (
                                                                        <button
                                                                            type="button"
                                                                            onClick={() => handleDeleteKey(p)}
                                                                            className="p-1 text-mute hover:text-error hover:bg-error-soft dark:hover:bg-[rgba(238,0,0,0.1)] rounded-v-sm transition"
                                                                            title={`Delete ${p.label} API key`}
                                                                            aria-label={`Delete ${p.label} API key`}
                                                                        >
                                                                            <Trash2 className="w-3.5 h-3.5" />
                                                                        </button>
                                                                    )}
                                                                </div>
                                                            </div>
                                                            <div className="relative">
                                                                <input
                                                                    type="password"
                                                                    className="input font-mono text-sm"
                                                                    placeholder={p.placeholder || "sk-..."}
                                                                    value={config[p.keyField] || ''}
                                                                    onChange={(e) => updateProviderKey(p.keyField, e.target.value)}
                                                                />
                                                                {config[`${p.keyField}_set`] && (
                                                                    <div className="absolute right-3 top-1/2 -translate-y-1/2 text-success" title="API key configured">
                                                                        <Check className="w-4 h-4" />
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        </div>
                                    )}

                                    {activeTab === 'external' && (
                                        <div className="space-y-6">
                                            <div>
                                                <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">External Providers</h3>
                                                <p className="text-sm text-body dark:text-[#888] mb-4">Connect to locally running Ollama or LM Studio servers.</p>
                                            </div>

                                            <div className="card p-4 space-y-4">
                                                <div>
                                                    <label className="label">Ollama Base URL</label>
                                                    <input
                                                        type="text"
                                                        className="input font-mono text-sm"
                                                        value={config.ollama_base_url || 'http://localhost:11434'}
                                                        onChange={(e) => setConfig({ ...config, ollama_base_url: e.target.value })}
                                                    />
                                                </div>
                                                <div>
                                                    <label className="label">LM Studio Base URL</label>
                                                    <input
                                                        type="text"
                                                        className="input font-mono text-sm"
                                                        value={config.lmstudio_base_url || 'http://localhost:1234/v1'}
                                                        onChange={(e) => setConfig({ ...config, lmstudio_base_url: e.target.value })}
                                                    />
                                                </div>
                                                <div>
                                                    <label className="label">Target Model Name (Optional)</label>
                                                    <input
                                                        type="text"
                                                        className="input font-mono text-sm"
                                                        placeholder="llama3"
                                                        value={config.external_model_name || ''}
                                                        onChange={(e) => setConfig({ ...config, external_model_name: e.target.value })}
                                                    />
                                                    <p className="text-[11px] text-mute mt-1">Model name/tag to request (e.g. the model loaded in LM Studio, or an Ollama tag). If left empty, the first model loaded on the server is used automatically — but LM Studio must have a model loaded, or requests fail with "model_not_found".</p>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {activeTab === 'local' && (
                                        <ModelManager
                                            activeModelPath={config.local_model_path}
                                            onSelectModel={(m) => setConfig({ ...config, local_model_path: m.path })}
                                        />
                                    )}

                                    {activeTab === 'appearance' && (
                                        <div className="space-y-6">
                                            <div>
                                                <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">Appearance & Theme</h3>
                                                <p className="text-sm text-body dark:text-[#888]">
                                                    Customize the visual theme and interface appearance of Docu AI Search.
                                                </p>
                                            </div>

                                            <div className="card p-5 space-y-4">
                                                <div>
                                                    <label className="label mb-2 block">Theme Preference</label>
                                                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                                                        {[
                                                            {
                                                                id: 'light',
                                                                label: 'Light',
                                                                icon: Sun,
                                                                description: 'Clean crisp layout with bright canvas backgrounds',
                                                            },
                                                            {
                                                                id: 'dark',
                                                                label: 'Dark',
                                                                icon: Moon,
                                                                description: 'Deep obsidian backgrounds engineered for low-light focus',
                                                            },
                                                            {
                                                                id: 'system',
                                                                label: 'System',
                                                                icon: Monitor,
                                                                description: 'Automatically synchronizes with your OS display mode',
                                                            },
                                                        ].map((t) => {
                                                            const isSelected = themePreference === t.id;
                                                            const Icon = t.icon;
                                                            return (
                                                                <button
                                                                    key={t.id}
                                                                    type="button"
                                                                    onClick={() => handleThemeChange(t.id)}
                                                                    className={`p-4 rounded-v-sm border text-left transition-all flex flex-col justify-between ${
                                                                        isSelected
                                                                            ? 'border-ink dark:border-[#ededed] bg-ink/5 dark:bg-white/10 ring-1 ring-ink dark:ring-[#ededed]'
                                                                            : 'border-hairline dark:border-[rgba(255,255,255,0.1)] bg-canvas-soft dark:bg-[rgba(255,255,255,0.03)] hover:border-ink/40 dark:hover:border-white/30'
                                                                    }`}
                                                                >
                                                                    <div className="flex items-center justify-between w-full mb-3">
                                                                        <div className={`p-2 rounded-v-sm ${isSelected ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink' : 'bg-canvas dark:bg-[rgba(255,255,255,0.08)] text-body dark:text-[#888]'}`}>
                                                                            <Icon className="w-4 h-4" />
                                                                        </div>
                                                                        {isSelected && (
                                                                            <span className="px-1.5 py-0.5 text-[10px] font-medium rounded-v-sm bg-ink/10 dark:bg-white/20 text-ink dark:text-[#ededed]">
                                                                                Active
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                    <div>
                                                                        <div className="font-semibold text-sm text-ink dark:text-[#ededed] mb-1">{t.label}</div>
                                                                        <div className="text-xs text-body dark:text-[#888] leading-relaxed">{t.description}</div>
                                                                    </div>
                                                                </button>
                                                            );
                                                        })}
                                                    </div>
                                                </div>

                                                <div className="pt-3 border-t border-hairline dark:border-[rgba(255,255,255,0.1)] flex items-center justify-between text-xs text-body dark:text-[#888]">
                                                    <span>Current active mode: <strong className="text-ink dark:text-[#ededed] font-medium">{darkMode ? 'Dark mode' : 'Light mode'}</strong></span>
                                                    <span className="font-mono text-[11px] opacity-75">Preference: {themePreference}</span>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {activeTab === 'system' && (
                                        <div className="space-y-6">
                                            <div>
                                                <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">System</h3>
                                                <p className="text-sm text-body dark:text-[#888] mb-4">Advanced configuration for indexing and routing.</p>
                                            </div>

                                            <div className="card p-4 space-y-4">
                                                <label className="flex items-start gap-3 cursor-pointer group">
                                                    <div className="mt-0.5">
                                                        <input
                                                            type="checkbox"
                                                            className="w-4 h-4 rounded border-hairline text-ink focus:ring-ink"
                                                            checked={config.query_rewriting}
                                                            onChange={(e) => setConfig({ ...config, query_rewriting: e.target.checked })}
                                                        />
                                                    </div>
                                                    <div>
                                                        <div className="font-medium text-ink dark:text-[#ededed] text-sm group-hover:text-link transition">Query Rewriting</div>
                                                        <div className="text-xs text-mute mt-0.5">Uses the active LLM to expand queries with synonyms before vector search.</div>
                                                    </div>
                                                </label>

                                                <label className="flex items-start gap-3 cursor-pointer group">
                                                    <div className="mt-0.5">
                                                        <input
                                                            type="checkbox"
                                                            className="w-4 h-4 rounded border-hairline text-ink focus:ring-ink"
                                                            checked={config.cross_encoder_reranking}
                                                            onChange={(e) => setConfig({ ...config, cross_encoder_reranking: e.target.checked })}
                                                        />
                                                    </div>
                                                    <div>
                                                        <div className="font-medium text-ink dark:text-[#ededed] text-sm group-hover:text-link transition">Cross-Encoder Re-ranking</div>
                                                        <div className="text-xs text-mute mt-0.5">Drastically improves relevance but increases search latency by 200-500ms.</div>
                                                    </div>
                                                </label>
                                            </div>

                                            <div className="card p-5">
                                                <div className="flex items-center justify-between mb-3">
                                                    <div>
                                                        <div className="font-medium text-sm text-ink dark:text-[#ededed]">AI response cache</div>
                                                        <div className="text-xs text-mute">Cached AI answers for repeated queries.</div>
                                                    </div>
                                                    <button onClick={clearCache} className="btn-secondary text-xs h-8">Clear cache</button>
                                                </div>
                                                <div className="flex gap-6">
                                                    <div>
                                                        <div className="text-2xl font-semibold text-ink dark:text-[#ededed] tracking-[-1px]">{cacheStats.total_entries}</div>
                                                        <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">Entries</div>
                                                    </div>
                                                    <div>
                                                        <div className="text-2xl font-semibold text-link tracking-[-1px]">{cacheStats.total_hits}</div>
                                                        <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">Total hits</div>
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="card p-5 flex items-center justify-between">
                                                <div>
                                                    <div className="font-medium text-sm text-ink dark:text-[#ededed]">Search history</div>
                                                    <div className="text-xs text-mute">Wipe every recorded search.</div>
                                                </div>
                                                <button
                                                    onClick={async () => {
                                                        if (!confirm('Clear all search history?')) return;
                                                        try {
                                                            await api.clearSearchHistory();
                                                            toast.success('History cleared');
                                                        } catch {
                                                            toast.error('Could not clear history');
                                                        }
                                                    }}
                                                    className="btn-danger text-xs h-8"
                                                >
                                                    Clear history
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Footer */}
                    <div className="px-6 py-4 border-t border-hairline dark:border-[rgba(255,255,255,0.08)] bg-canvas-soft dark:bg-[#0a0a0a] flex justify-end gap-3">
                        <button onClick={onClose} className="btn-secondary" disabled={loading || saving}>
                            Cancel
                        </button>
                        <button onClick={save} className="btn-primary" disabled={loading || saving}>
                            {saving ? 'Saving...' : 'Save Changes'}
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
}
