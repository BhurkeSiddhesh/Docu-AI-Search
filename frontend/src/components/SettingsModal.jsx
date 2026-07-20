import React, { useEffect, useState } from 'react';
import { X, Folder, Link2, Check, Settings, Trash2, Shield, Cloud, Cpu, Sparkles, Layers, History } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';
import ModelManager from './ModelManager';

const PROVIDERS = [
    { id: 'openai', label: 'OpenAI', icon: Cloud, keyField: 'openai_api_key' },
    { id: 'gemini', label: 'Google Gemini', icon: Sparkles, keyField: 'gemini_api_key' },
    { id: 'anthropic', label: 'Anthropic Claude', icon: Cloud, keyField: 'anthropic_api_key' },
    { id: 'grok', label: 'xAI Grok', icon: Cloud, keyField: 'grok_api_key' },
];

export default function SettingsModal({ isOpen, onClose, onSaved }) {
    const [config, setConfig] = useState(null);
    const [embeddingConfig, setEmbeddingConfig] = useState(null);
    const [cacheStats, setCacheStats] = useState({ total_entries: 0, total_hits: 0 });
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [activeTab, setActiveTab] = useState('folders');
    const [newFolder, setNewFolder] = useState('');
    const [folderHistory, setFolderHistory] = useState([]);
    const [showFolderHistory, setShowFolderHistory] = useState(false);
    const [validatingFolder, setValidatingFolder] = useState(false);
    const toast = useToast();

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
            const [cRes, eRes, hRes, cacheRes] = await Promise.all([
                api.getConfig(),
                api.getEmbeddingConfig(),
                api.getFolderHistory(),
                api.getCacheStats()
            ]);
            setConfig(cRes.data);
            setEmbeddingConfig(eRes.data);
            setFolderHistory(hRes.data || []);
            setCacheStats(cacheRes.data);
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

                                    {activeTab === 'embeddings' && (
                                        <div className="space-y-6">
                                            <div>
                                                <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">Embeddings</h3>
                                                <p className="text-sm text-body dark:text-[#888] mb-4">Embeddings turn text into vectors for semantic search. Changing the model requires re-indexing.</p>
                                            </div>
                                            
                                            <div className="card p-4 space-y-4">
                                                <div className="flex gap-2">
                                                    <button
                                                        onClick={() => setEmbeddingConfig({ ...embeddingConfig, provider_type: 'local' })}
                                                        className={`flex-1 py-2 rounded-v-sm text-sm transition ${embeddingConfig.provider_type === 'local' ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink' : 'bg-canvas-soft border border-hairline text-body hover:bg-canvas'}`}
                                                    >
                                                        Local (on-device)
                                                    </button>
                                                    <button
                                                        onClick={() => setEmbeddingConfig({ ...embeddingConfig, provider_type: 'huggingface_api' })}
                                                        className={`flex-1 py-2 rounded-v-sm text-sm transition ${embeddingConfig.provider_type === 'huggingface_api' ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink' : 'bg-canvas-soft border border-hairline text-body hover:bg-canvas'}`}
                                                    >
                                                        HuggingFace API
                                                    </button>
                                                </div>

                                                <div>
                                                    <label htmlFor="model_name" className="label">Model name</label>
                                                    <input
                                                        id="model_name"
                                                        type="text"
                                                        className="input font-mono text-sm"
                                                        value={embeddingConfig.model_name || ''}
                                                        onChange={(e) => setEmbeddingConfig({ ...embeddingConfig, model_name: e.target.value })}
                                                    />
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
                                        </div>
                                    )}

                                    {activeTab === 'providers' && (
                                        <div className="space-y-6">
                                            <div>
                                                <h3 className="text-display-sm text-ink dark:text-[#ededed] mb-1">Cloud Providers</h3>
                                                <p className="text-sm text-body dark:text-[#888] mb-4">Enter API keys to enable cloud-based LLM generation. Keys are stored locally.</p>
                                            </div>
                                            
                                            <div className="space-y-4">
                                                {PROVIDERS.map(p => (
                                                    <div key={p.id} className="card p-4">
                                                        <label className="label flex items-center gap-2 mb-3 text-ink dark:text-[#ededed] font-sans font-medium tracking-normal text-sm normal-case">
                                                            <p.icon className="w-4 h-4" />
                                                            {p.label}
                                                        </label>
                                                        <div className="relative">
                                                            <input
                                                                type="password"
                                                                className="input font-mono text-sm"
                                                                placeholder="sk-..."
                                                                value={config[p.keyField] || ''}
                                                                onChange={(e) => updateProviderKey(p.keyField, e.target.value)}
                                                            />
                                                            {config[`${p.keyField}_set`] && (
                                                                <div className="absolute right-3 top-1/2 -translate-y-1/2 text-success">
                                                                    <Check className="w-4 h-4" />
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                ))}
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
                                                    <p className="text-[11px] text-mute mt-1">Leave empty to use the provider's default model.</p>
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
