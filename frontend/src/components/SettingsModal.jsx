import React, { useEffect, useState } from 'react';
import { X, FolderOpen, Cloud, Cpu, Database, Server, Trash2, Plus, Settings as SettingsIcon, RefreshCw, History, Layers } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';
import ModelManager from './ModelManager';

const SECTIONS = [
    { id: 'folders',     label: 'Folders',    icon: FolderOpen },
    { id: 'embeddings',  label: 'Embeddings', icon: Layers },
    { id: 'providers',   label: 'Cloud LLM',  icon: Cloud },
    { id: 'external',    label: 'External',   icon: Server },
    { id: 'local',       label: 'Local LLM',  icon: Cpu },
    { id: 'system',      label: 'System',     icon: Database },
];

const EMBEDDING_TYPES = [
    { value: 'local',           label: 'Local (on-device)',     needsKey: false },
    { value: 'huggingface_api', label: 'HuggingFace API',         needsKey: true  },
    { value: 'commercial_api',  label: 'Cloud (OpenAI / Gemini)', needsKey: true  },
];

const DEFAULT_EMBEDDING = {
    provider_type: 'local',
    model_name: 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    api_key: '',
};

export default function SettingsModal({ isOpen, onClose, onSaved }) {
    const [section, setSection] = useState('folders');
    const [config, setConfig] = useState(null);
    const [embedding, setEmbedding] = useState(DEFAULT_EMBEDDING);
    const [folderHistory, setFolderHistory] = useState([]);
    const [cacheStats, setCacheStats] = useState({ total_entries: 0, total_hits: 0 });
    const [saving, setSaving] = useState(false);
    const [pathInput, setPathInput] = useState('');
    const [pathValidating, setPathValidating] = useState(false);
    const [pathInfo, setPathInfo] = useState(null);
    const toast = useToast();

    useEffect(() => {
        if (!isOpen) return;
        loadAll();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') onClose();
        };
        if (isOpen) {
            window.addEventListener('keydown', handleKeyDown);
        }
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [isOpen, onClose]);

    const loadAll = async () => {
        try {
            const [c, e, h, cs] = await Promise.all([
                api.getConfig().catch(() => ({ data: {} })),
                api.getEmbeddingConfig().catch(() => ({ data: DEFAULT_EMBEDDING })),
                api.getFolderHistory().catch(() => ({ data: [] })),
                api.getCacheStats().catch(() => ({ data: { total_entries: 0, total_hits: 0 } })),
            ]);
            const cfg = c.data || {};
            setConfig({
                folders:           cfg.folders || [],
                auto_index:        cfg.auto_index || false,
                provider:          cfg.provider || 'openai',
                local_model_path:  cfg.local_model_path || '',
                tensor_split:      cfg.tensor_split || '',
                openai_api_key:    '',
                gemini_api_key:    '',
                anthropic_api_key: '',
                grok_api_key:      '',
                openai_api_key_set:    cfg.openai_api_key_set,
                gemini_api_key_set:    cfg.gemini_api_key_set,
                anthropic_api_key_set: cfg.anthropic_api_key_set,
                grok_api_key_set:      cfg.grok_api_key_set,
                query_rewriting:          cfg.query_rewriting || false,
                cross_encoder_reranking:  cfg.cross_encoder_reranking || false,
                reranker_model:           cfg.reranker_model || 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                ollama_base_url:    cfg.ollama_base_url || 'http://localhost:11434',
                lmstudio_base_url:  cfg.lmstudio_base_url || 'http://localhost:1234/v1',
                external_model_name: cfg.external_model_name || '',
                external_api_key:   cfg.external_api_key || '',
            });
            setEmbedding({
                provider_type: e.data.provider_type || 'local',
                model_name:    e.data.model_name || DEFAULT_EMBEDDING.model_name,
                api_key:       e.data.api_key_set ? '••••••••' : '',
            });
            setFolderHistory(h.data || []);
            setCacheStats(cs.data || { total_entries: 0, total_hits: 0 });
        } catch (err) {
            toast.error('Failed to load settings');
        }
    };

    const save = async () => {
        if (!config) return;
        setSaving(true);
        try {
            await api.saveConfig({
                folders:           config.folders,
                auto_index:        config.auto_index,
                provider:          config.provider,
                local_model_path:  config.local_model_path,
                tensor_split:      config.tensor_split || null,
                openai_api_key:    config.openai_api_key,
                gemini_api_key:    config.gemini_api_key,
                anthropic_api_key: config.anthropic_api_key,
                grok_api_key:      config.grok_api_key,
                query_rewriting:         config.query_rewriting,
                cross_encoder_reranking: config.cross_encoder_reranking,
                reranker_model:          config.reranker_model,
                ollama_base_url:     config.ollama_base_url,
                lmstudio_base_url:   config.lmstudio_base_url,
                external_model_name: config.external_model_name,
                external_api_key:    config.external_api_key,
            });
            const embPayload = {
                provider_type: embedding.provider_type,
                model_name:    embedding.model_name,
            };
            if (embedding.api_key && embedding.api_key !== '••••••••') {
                embPayload.api_key = embedding.api_key;
            }
            try {
                await api.saveEmbeddingConfig(embPayload);
            } catch {
                // embedding endpoint may not exist on all builds — non-fatal
            }
            toast.success('Settings saved');
            onSaved?.();
            onClose();
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Save failed');
        } finally {
            setSaving(false);
        }
    };

    const persistFolders = async (nextFolders) => {
        try {
            await api.saveConfig({
                folders:           nextFolders,
                auto_index:        config.auto_index,
                provider:          config.provider,
                local_model_path:  config.local_model_path,
                tensor_split:      config.tensor_split || null,
                openai_api_key:    '',
                gemini_api_key:    '',
                anthropic_api_key: '',
                grok_api_key:      '',
                query_rewriting:         config.query_rewriting,
                cross_encoder_reranking: config.cross_encoder_reranking,
                reranker_model:          config.reranker_model,
                ollama_base_url:     config.ollama_base_url,
                lmstudio_base_url:   config.lmstudio_base_url,
                external_model_name: config.external_model_name,
                external_api_key:    config.external_api_key,
            });
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not save folder');
        }
    };

    const addFolder = async (path) => {
        const p = (path || '').trim();
        if (!p) return;
        if (config.folders.includes(p)) {
            toast.info('Folder already added');
            return;
        }
        const next = [...config.folders, p];
        setConfig((c) => ({ ...c, folders: next }));
        setPathInput('');
        setPathInfo(null);
        await persistFolders(next);
    };

    const removeFolder = async (p) => {
        const next = config.folders.filter((x) => x !== p);
        setConfig((c) => ({ ...c, folders: next }));
        await persistFolders(next);
    };

    const browseFolder = async () => {
        try {
            const r = await api.browse();
            if (r.data.folder) addFolder(r.data.folder);
        } catch (e) {
            toast.error('Folder browser unavailable. Paste the path manually.');
        }
    };

    const validatePath = async () => {
        if (!pathInput.trim()) return;
        setPathValidating(true);
        try {
            const r = await api.validatePath(pathInput.trim());
            setPathInfo(r.data);
        } catch (e) {
            setPathInfo({ valid: false, error: 'Validation failed' });
        } finally {
            setPathValidating(false);
        }
    };

    const startIndexing = async () => {
        try {
            await api.startIndexing();
            toast.success('Indexing started');
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not start indexing');
        }
    };

    const clearCache = async () => {
        try {
            await api.clearCache();
            const r = await api.getCacheStats();
            setCacheStats(r.data);
            toast.success('Cache cleared');
        } catch {
            toast.error('Could not clear cache');
        }
    };
    if (!isOpen) return null;
    return (
        <div
            className="fixed inset-0 z-[80] bg-slate-900/50 flex items-end sm:items-center justify-center p-0 sm:p-4 animate-fade-in"
            onClick={(e) => e.target === e.currentTarget && onClose()}
            onKeyDown={(e) => {
                if (e.key === 'Escape') onClose();
            }}
            tabIndex={-1}
        >
            <div 
                className="bg-white dark:bg-slate-900 w-full sm:max-w-5xl h-[92vh] sm:h-[88vh] sm:rounded-xl border border-slate-200 dark:border-slate-800 shadow-2xl flex flex-col overflow-hidden"
                role="dialog"
                aria-modal="true"
                aria-labelledby="settings-modal-title"
            >
                {/* Header */}
                <header className="flex items-center justify-between px-5 py-4 border-b border-slate-200 dark:border-slate-800">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
                            <SettingsIcon className="w-4 h-4" />
                        </div>
                        <h2 id="settings-modal-title" className="font-semibold text-slate-900 dark:text-slate-50">Settings</h2>
                    </div>
                    <button onClick={onClose} className="p-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800" aria-label="Close settings">
                        <X className="w-5 h-5" />
                    </button>
                </header>

                <div className="flex-1 flex flex-col sm:flex-row overflow-hidden min-h-0">
                    {/* Tab nav */}
                    <nav className="sm:w-52 border-b sm:border-b-0 sm:border-r border-slate-200 dark:border-slate-800 p-2 flex sm:flex-col gap-1 overflow-x-auto sm:overflow-y-auto no-scrollbar flex-shrink-0">
                        {SECTIONS.map((s) => {
                            const Icon = s.icon;
                            const active = section === s.id;
                            return (
                                <button
                                    key={s.id}
                                    onClick={() => setSection(s.id)}
                                    className={`flex-shrink-0 sm:flex-shrink flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition whitespace-nowrap ${
                                        active
                                            ? 'bg-primary/10 text-primary'
                                            : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                                    }`}
                                >
                                    <Icon className="w-4 h-4" />
                                    {s.label}
                                </button>
                            );
                        })}
                    </nav>

                    {/* Body */}
                    <main className="flex-1 overflow-y-auto p-5 sm:p-6 min-h-0">
                        {!config ? (
                            <div className="flex items-center justify-center h-full text-slate-500 dark:text-slate-400 text-sm">
                                Loading…
                            </div>
                        ) : (
                            <>
                                {section === 'folders' && (
                                    <div className="space-y-5 max-w-2xl">
                                        <div>
                                            <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-1">Indexed folders</h3>
                                            <p className="text-sm text-slate-500 dark:text-slate-400">Add the folders you want the search engine to read from.</p>
                                        </div>

                                        <div className="card p-4">
                                            <div className="label">Add folder</div>
                                            <div className="flex flex-col sm:flex-row gap-2">
                                                <input
                                                    value={pathInput}
                                                    onChange={(e) => { setPathInput(e.target.value); setPathInfo(null); }}
                                                    placeholder="C:\\Users\\you\\Documents"
                                                    className="input font-mono text-xs"
                                                />
                                                <div className="flex gap-2">
                                                    <button onClick={validatePath} disabled={pathValidating || !pathInput.trim()} className="btn-secondary text-xs">
                                                        {pathValidating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : 'Validate'}
                                                    </button>
                                                    <button onClick={browseFolder} className="btn-secondary text-xs">
                                                        <FolderOpen className="w-3.5 h-3.5" />
                                                        Browse
                                                    </button>
                                                </div>
                                            </div>
                                            {pathInfo && (
                                                <div className={`mt-2 text-xs px-3 py-2 rounded-md ${
                                                    pathInfo.valid
                                                        ? 'bg-green-50 dark:bg-green-950/40 text-green-700 dark:text-green-300'
                                                        : 'bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300'
                                                }`}>
                                                    {pathInfo.valid
                                                        ? `Found ${pathInfo.file_count} supported file${pathInfo.file_count === 1 ? '' : 's'}.`
                                                        : pathInfo.error}
                                                </div>
                                            )}
                                            {pathInfo?.valid && (
                                                <button onClick={() => addFolder(pathInput)} className="btn-primary mt-2 text-xs">
                                                    <Plus className="w-3.5 h-3.5" />
                                                    Add to library
                                                </button>
                                            )}
                                        </div>

                                        <div className="space-y-2">
                                            {config.folders.length === 0 ? (
                                                <div className="card p-6 text-center text-sm text-slate-500 dark:text-slate-400">
                                                    No folders added yet.
                                                </div>
                                            ) : (
                                                config.folders.map((f) => (
                                                    <div key={f} className="card p-3 flex items-center gap-3">
                                                        <FolderOpen className="w-4 h-4 text-primary flex-shrink-0" />
                                                        <span className="font-mono text-xs flex-1 truncate" title={f}>{f}</span>
                                                        <button
                                                            onClick={() => removeFolder(f)}
                                                            className="p-1.5 rounded-md text-slate-500 hover:bg-red-50 dark:hover:bg-red-950/40 hover:text-red-500 transition"
                                                            aria-label={`Remove ${f} from index`}
                                                        >
                                                            <Trash2 className="w-3.5 h-3.5" />
                                                        </button>
                                                    </div>
                                                ))
                                            )}
                                        </div>

                                        <button onClick={startIndexing} className="btn-primary">
                                            <RefreshCw className="w-4 h-4" />
                                            Start indexing
                                        </button>

                                        {folderHistory.length > 0 && (
                                            <div>
                                                <div className="label flex items-center gap-1.5 mt-4">
                                                    <History className="w-3.5 h-3.5" />
                                                    Previously indexed
                                                </div>
                                                <div className="space-y-1.5">
                                                    {folderHistory.map((p) => (
                                                        <div key={p} className="flex items-center gap-2 group">
                                                            <span className="text-xs font-mono text-slate-500 dark:text-slate-400 truncate flex-1" title={p}>{p}</span>
                                                            {!config.folders.includes(p) && (
                                                                <button
                                                                    onClick={() => addFolder(p)}
                                                                    className="text-xs font-medium text-primary hover:underline opacity-0 group-hover:opacity-100"
                                                                >
                                                                    Restore
                                                                </button>
                                                            )}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}

                                {section === 'embeddings' && (
                                    <div className="space-y-5 max-w-2xl">
                                        <div>
                                            <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-1">Embeddings</h3>
                                            <p className="text-sm text-slate-500 dark:text-slate-400">
                                                Embeddings turn text into vectors for semantic search. Changing the model requires re-indexing.
                                            </p>
                                        </div>

                                        <div className="grid sm:grid-cols-3 gap-2">
                                            {EMBEDDING_TYPES.map((t) => (
                                                <button
                                                    key={t.value}
                                                    onClick={() => setEmbedding((s) => ({ ...s, provider_type: t.value }))}
                                                    className={`p-3 rounded-lg border text-left transition ${
                                                        embedding.provider_type === t.value
                                                            ? 'border-primary bg-primary/5 text-primary'
                                                            : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                                                    }`}
                                                >
                                                    <Layers className="w-4 h-4 mb-2" />
                                                    <div className="text-sm font-medium">{t.label}</div>
                                                </button>
                                            ))}
                                        </div>

                                        <div>
                                            <label htmlFor="embedding-model-name" className="label">Model name</label>
                                            <input
                                                id="embedding-model-name"
                                                className="input font-mono text-xs"
                                                value={embedding.model_name}
                                                onChange={(e) => setEmbedding((s) => ({ ...s, model_name: e.target.value }))}
                                            />
                                        </div>
 
                                        {EMBEDDING_TYPES.find((t) => t.value === embedding.provider_type)?.needsKey && (
                                            <div>
                                                <label htmlFor="embedding-api-key" className="label">API key</label>
                                                <input
                                                    id="embedding-api-key"
                                                    type="password"
                                                    className="input"
                                                    value={embedding.api_key}
                                                    onChange={(e) => setEmbedding((s) => ({ ...s, api_key: e.target.value }))}
                                                    placeholder="Enter API key"
                                                />
                                            </div>
                                        )}
                                    </div>
                                )}

                                {section === 'providers' && (
                                    <div className="space-y-5 max-w-2xl">
                                        <div>
                                            <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-1">Cloud providers</h3>
                                            <p className="text-sm text-slate-500 dark:text-slate-400">API keys are stored locally in config.ini. Leave a field blank to keep the existing key.</p>
                                        </div>

                                        <div>
                                            <div className="label">Active provider</div>
                                            <select
                                                className="input"
                                                value={config.provider}
                                                onChange={(e) => setConfig((c) => ({ ...c, provider: e.target.value }))}
                                            >
                                                <option value="local">Local (GGUF)</option>
                                                <option value="openai">OpenAI</option>
                                                <option value="gemini">Google Gemini</option>
                                                <option value="anthropic">Anthropic Claude</option>
                                                <option value="grok">xAI Grok</option>
                                                <option value="ollama">Ollama</option>
                                                <option value="lmstudio">LM Studio</option>
                                            </select>
                                        </div>

                                        {[
                                            { id: 'openai_api_key',    label: 'OpenAI API key',    setKey: 'openai_api_key_set' },
                                            { id: 'gemini_api_key',    label: 'Gemini API key',    setKey: 'gemini_api_key_set' },
                                            { id: 'anthropic_api_key', label: 'Anthropic API key', setKey: 'anthropic_api_key_set' },
                                            { id: 'grok_api_key',      label: 'Grok API key',      setKey: 'grok_api_key_set' },
                                        ].map((p) => (
                                            <div key={p.id}>
                                                <div className="label flex items-center justify-between">
                                                    <span>{p.label}</span>
                                                    {config[p.setKey] && (
                                                        <span className="text-[10px] font-medium normal-case tracking-normal text-green-600 dark:text-green-400">
                                                            Key configured
                                                        </span>
                                                    )}
                                                </div>
                                                <input
                                                    id={p.id}
                                                    type="password"
                                                    className="input"
                                                    value={config[p.id] || ''}
                                                    onChange={(e) => setConfig((c) => ({ ...c, [p.id]: e.target.value }))}
                                                    placeholder={config[p.setKey] ? 'Leave blank to keep existing key' : 'Paste API key'}
                                                />
                                            </div>
                                        ))}

                                        <div className="card p-4 space-y-3">
                                            <div className="text-sm font-medium">Advanced RAG</div>
                                            <label className="flex items-center gap-3 cursor-pointer">
                                                <input
                                                    type="checkbox"
                                                    checked={config.query_rewriting}
                                                    onChange={(e) => setConfig((c) => ({ ...c, query_rewriting: e.target.checked }))}
                                                />
                                                <span className="text-sm">Enable LLM query rewriting</span>
                                            </label>
                                            <label className="flex items-center gap-3 cursor-pointer">
                                                <input
                                                    type="checkbox"
                                                    checked={config.cross_encoder_reranking}
                                                    onChange={(e) => setConfig((c) => ({ ...c, cross_encoder_reranking: e.target.checked }))}
                                                />
                                                <span className="text-sm">Enable cross-encoder reranking</span>
                                            </label>
                                        </div>
                                    </div>
                                )}

                                {section === 'external' && (
                                    <div className="space-y-5 max-w-2xl">
                                        <div>
                                            <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-1">External servers</h3>
                                            <p className="text-sm text-slate-500 dark:text-slate-400">Connect to Ollama or LM Studio running on your machine or network.</p>
                                        </div>

                                        <div>
                                            <div className="label">Ollama base URL</div>
                                            <input
                                                className="input font-mono text-xs"
                                                value={config.ollama_base_url}
                                                onChange={(e) => setConfig((c) => ({ ...c, ollama_base_url: e.target.value }))}
                                            />
                                        </div>
                                        <div>
                                            <div className="label">LM Studio base URL</div>
                                            <input
                                                className="input font-mono text-xs"
                                                value={config.lmstudio_base_url}
                                                onChange={(e) => setConfig((c) => ({ ...c, lmstudio_base_url: e.target.value }))}
                                            />
                                        </div>
                                        <div>
                                            <div className="label">External model name</div>
                                            <input
                                                className="input"
                                                value={config.external_model_name}
                                                onChange={(e) => setConfig((c) => ({ ...c, external_model_name: e.target.value }))}
                                                placeholder="e.g. llama3.1:8b"
                                            />
                                        </div>
                                        <div>
                                            <div className="label">External API key (optional)</div>
                                            <input
                                                type="password"
                                                className="input"
                                                value={config.external_api_key}
                                                onChange={(e) => setConfig((c) => ({ ...c, external_api_key: e.target.value }))}
                                            />
                                        </div>
                                    </div>
                                )}

                                {section === 'local' && (
                                    <div className="max-w-3xl">
                                        <div className="mb-4">
                                            <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-1">Local LLM models</h3>
                                            <p className="text-sm text-slate-500 dark:text-slate-400">Download GGUF models for fully offline inference.</p>
                                        </div>
                                        <ModelManager
                                            activeModelPath={config.local_model_path}
                                            onSelectModel={(m) => setConfig((c) => ({ ...c, local_model_path: m.path }))}
                                        />
                                    </div>
                                )}

                                {section === 'system' && (
                                    <div className="space-y-5 max-w-2xl">
                                        <div>
                                            <h3 className="font-semibold text-slate-900 dark:text-slate-50 mb-1">System</h3>
                                            <p className="text-sm text-slate-500 dark:text-slate-400">Manage caches and history.</p>
                                        </div>

                                        <div className="card p-5">
                                            <div className="flex items-center justify-between mb-3">
                                                <div>
                                                    <div className="font-semibold text-sm text-slate-900 dark:text-slate-50">AI response cache</div>
                                                    <div className="text-xs text-slate-500 dark:text-slate-400">Cached AI answers for repeated queries.</div>
                                                </div>
                                                <button onClick={clearCache} className="btn-secondary text-xs">Clear cache</button>
                                            </div>
                                            <div className="flex gap-6">
                                                <div>
                                                    <div className="text-2xl font-bold text-slate-900 dark:text-slate-50">{cacheStats.total_entries}</div>
                                                    <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Entries</div>
                                                </div>
                                                <div>
                                                    <div className="text-2xl font-bold text-primary">{cacheStats.total_hits}</div>
                                                    <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Total hits</div>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="card p-5">
                                            <div className="flex items-center justify-between">
                                                <div>
                                                    <div className="font-semibold text-sm text-slate-900 dark:text-slate-50">Search history</div>
                                                    <div className="text-xs text-slate-500 dark:text-slate-400">Wipe every recorded search.</div>
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
                                                    className="btn-danger text-xs"
                                                >
                                                    Clear history
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
                <footer className="flex items-center justify-end gap-2 px-5 py-3 border-t border-slate-200 dark:border-slate-800">
                    <button onClick={onClose} className="btn-ghost">Cancel</button>
                    <button onClick={save} disabled={saving || !config} className="btn-primary">
                        {saving ? 'Saving…' : 'Save Changes'}
                    </button>
                </footer>
            </div>
        </div>
    );
}
