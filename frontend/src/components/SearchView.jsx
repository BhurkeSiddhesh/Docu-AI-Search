import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Search, Sparkles, Loader2, Bot, ChevronDown, Filter, Cpu, Cloud, Zap } from 'lucide-react';
import ResultCard from './ResultCard';
import AgentView from './AgentView';
import api from '../lib/api';
import { useToast } from './Toast';

const FILE_TYPES = ['pdf', 'docx', 'xlsx', 'csv', 'pptx', 'txt', 'md'];
const SORT_OPTIONS = [
    { value: 'relevance', label: 'Relevance' },
    { value: 'date',      label: 'Date modified' },
    { value: 'filename',  label: 'Filename' },
    { value: 'file_size', label: 'File size' },
];

const CLOUD_PROVIDERS = [
    { id: 'openai',    label: 'OpenAI',          keyField: 'openai_api_key_set' },
    { id: 'gemini',    label: 'Google Gemini',    keyField: 'gemini_api_key_set' },
    { id: 'anthropic', label: 'Anthropic Claude', keyField: 'anthropic_api_key_set' },
    { id: 'grok',      label: 'xAI Grok',        keyField: 'grok_api_key_set' },
];

export default function SearchView({ pendingQuery }) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [aiAnswer, setAiAnswer] = useState('');
    const [activeModel, setActiveModel] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [error, setError] = useState(null);

    const [agentMode, setAgentMode] = useState(false);
    const [agentQuery, setAgentQuery] = useState('');

    const [showFilters, setShowFilters] = useState(false);
    const [filters, setFilters] = useState({
        file_types: [],
        sort_by: 'relevance',
        min_score: null,
    });

    // Model selector state
    const [showModelPicker, setShowModelPicker] = useState(false);
    const [modelConfig, setModelConfig] = useState(null);
    const [localModels, setLocalModels] = useState([]);
    const [selectedProvider, setSelectedProvider] = useState('');
    const [selectedModelPath, setSelectedModelPath] = useState('');
    const modelPickerRef = useRef(null);

    const inputRef = useRef(null);
    const streamAbortRef = useRef(null);
    const toast = useToast();

    // Load config + models for the model picker
    const loadModelOptions = useCallback(async () => {
        try {
            const [c, m] = await Promise.all([
                api.getConfig(),
                api.listLocalModels().catch(() => ({ data: [] })),
            ]);
            const cfg = c.data || {};
            setModelConfig(cfg);
            setLocalModels(m.data || []);
            setSelectedProvider(cfg.provider || 'openai');
            setSelectedModelPath(cfg.local_model_path || '');
        } catch {
            // silent — model picker will be hidden if no config
        }
    }, []);

    useEffect(() => {
        loadModelOptions();
    }, [loadModelOptions]);

    // Close model picker on outside click
    useEffect(() => {
        const handler = (e) => {
            if (modelPickerRef.current && !modelPickerRef.current.contains(e.target)) {
                setShowModelPicker(false);
            }
        };
        if (showModelPicker) {
            document.addEventListener('mousedown', handler);
        }
        return () => document.removeEventListener('mousedown', handler);
    }, [showModelPicker]);

    useEffect(() => {
        // Cancel any in-flight answer stream when the view unmounts
        return () => streamAbortRef.current?.abort();
    }, []);

    useEffect(() => {
        // External query trigger (e.g., from history). Re-runs whenever the
        // pendingQuery object identity changes, even for the same query string.
        if (pendingQuery?.q) {
            setQuery(pendingQuery.q);
            runSearch(pendingQuery.q);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pendingQuery]);

    useEffect(() => {
        const handler = (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                inputRef.current?.focus();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    const selectModel = async (provider, modelPath = '') => {
        setSelectedProvider(provider);
        setSelectedModelPath(modelPath);
        setShowModelPicker(false);
        try {
            await api.saveConfig({
                ...buildConfigPayload(),
                provider,
                local_model_path: modelPath,
            });
            toast.success(`Switched to ${getModelLabel(provider, modelPath)}`);
        } catch {
            toast.error('Could not switch model');
        }
    };

    const buildConfigPayload = () => {
        if (!modelConfig) return {};
        return {
            folders: modelConfig.folders || [],
            auto_index: modelConfig.auto_index || false,
            provider: selectedProvider,
            local_model_path: selectedModelPath,
            tensor_split: modelConfig.tensor_split || null,
            openai_api_key: '',
            gemini_api_key: '',
            anthropic_api_key: '',
            grok_api_key: '',
            query_rewriting: modelConfig.query_rewriting || false,
            cross_encoder_reranking: modelConfig.cross_encoder_reranking || false,
            reranker_model: modelConfig.reranker_model || '',
            ollama_base_url: modelConfig.ollama_base_url || 'http://localhost:11434',
            lmstudio_base_url: modelConfig.lmstudio_base_url || 'http://localhost:1234/v1',
            external_model_name: modelConfig.external_model_name || '',
            external_api_key: modelConfig.external_api_key || '',
        };
    };

    const getModelLabel = (provider, modelPath) => {
        if (provider === 'local' && modelPath) {
            const name = modelPath.split(/[/\\]/).pop();
            return name?.replace(/\.gguf$/i, '') || 'Local model';
        }
        const found = CLOUD_PROVIDERS.find((p) => p.id === provider);
        if (found) return found.label;
        if (provider === 'ollama') return 'Ollama';
        if (provider === 'lmstudio') return 'LM Studio';
        return provider;
    };

    const currentModelLabel = getModelLabel(selectedProvider, selectedModelPath);

    const norm = (s) => (s || '').replace(/\\/g, '/').toLowerCase();

    const runSearch = async (q) => {
        if (!q.trim()) return;
        // Cancel a previous, still-streaming answer so its tokens can't
        // interleave with (or overwrite) this search's answer.
        streamAbortRef.current?.abort();
        setError(null);
        setHasSearched(true);
        setResults([]);
        setAiAnswer('');

        if (agentMode) {
            setAgentQuery(q);
            return;
        }

        setIsSearching(true);
        try {
            const payload = {
                query: q,
                file_types: filters.file_types.length ? filters.file_types : undefined,
                sort_by: filters.sort_by !== 'relevance' ? filters.sort_by : undefined,
                min_score: filters.min_score ?? undefined,
            };
            const res = await api.search(q, payload);
            setResults(res.data.results || []);
            setActiveModel(res.data.active_model || '');
            setIsSearching(false);

            if ((res.data.results || []).length > 0) {
                // Stream the AI answer
                const controller = new AbortController();
                streamAbortRef.current = controller;
                setIsStreaming(true);
                let acc = '';
                try {
                    // Keep the LLM prompt small: on CPU every extra snippet adds
                    // seconds before the first token appears.
                    const context = res.data.results.map((r) => r.summary || r.document).slice(0, 4);
                    await api.streamAnswer(q, context, null, (chunk) => {
                        if (controller.signal.aborted) return;
                        acc += chunk;
                        setAiAnswer(acc);
                    }, controller.signal);
                } catch (e) {
                    if (e?.name !== 'AbortError') {
                        console.error('Stream error:', e);
                        toast.error('AI answer stream failed');
                    }
                } finally {
                    // Only clear the spinner if a newer search hasn't taken over
                    if (streamAbortRef.current === controller) {
                        setIsStreaming(false);
                    }
                }
            }
        } catch (e) {
            setIsSearching(false);
            const msg = e.response?.data?.detail || e.message || 'Search failed';
            setError(msg);
            if (e.response?.status === 400) {
                toast.error('No index yet. Configure and index a folder in Settings.');
            } else {
                toast.error(msg);
            }
        }
    };

    const onSubmit = (e) => {
        e.preventDefault();
        runSearch(query);
    };

    const toggleFileType = (t) => {
        setFilters((f) => ({
            ...f,
            file_types: f.file_types.includes(t)
                ? f.file_types.filter((x) => x !== t)
                : [...f.file_types, t],
        }));
    };

    return (
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 lg:py-10">
            {/* Hero (only before first search) */}
            {!hasSearched && (
                <div className="text-center mb-12 mt-4 animate-fade-in">
                    {/* Mesh gradient backdrop */}
                    <div className="relative mx-auto mb-8 w-full max-w-md h-40 rounded-v-lg overflow-hidden">
                        <div className="absolute inset-0 mesh-gradient" />
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-16 h-16 rounded-v-md bg-canvas dark:bg-[#171717] shadow-v-4 dark:shadow-v-dark-4 flex items-center justify-center">
                                <Search className="w-7 h-7 text-ink dark:text-[#ededed]" />
                            </div>
                        </div>
                    </div>
                    <h1 className="text-display-xl text-ink dark:text-[#ededed] mb-3">
                        Search your documents.
                    </h1>
                    <p className="text-body dark:text-[#888] text-lg leading-7">
                        Ask natural language questions across your indexed files.
                    </p>
                </div>
            )}

            {/* Search bar */}
            <form onSubmit={onSubmit}>
                <div className="relative">
                    <div className="absolute left-4 top-1/2 -translate-y-1/2 text-mute">
                        <Search className="w-5 h-5" />
                    </div>
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder={agentMode ? 'Ask the research agent...' : 'Search your documents... (Ctrl+K)'}
                        disabled={isSearching}
                        className="w-full bg-canvas dark:bg-[#0a0a0a] border border-hairline dark:border-[rgba(255,255,255,0.15)] rounded-v-sm pl-12 pr-36 h-12 text-[15px] text-ink dark:text-[#ededed] placeholder:text-mute outline-none transition-all focus:border-ink dark:focus:border-[#ededed] focus:shadow-[0_0_0_1px_#171717] dark:focus:shadow-[0_0_0_1px_#ededed]"
                        aria-label="Search"
                    />
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
                        <button
                            type="button"
                            onClick={() => setAgentMode(!agentMode)}
                            title={agentMode ? 'Agent mode on' : 'Enable agent mode'}
                            className={`p-2 rounded-v-sm transition ${
                                agentMode
                                    ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink'
                                    : 'text-mute hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]'
                            }`}
                        >
                            <Bot className="w-4 h-4" />
                        </button>
                        <button
                            type="submit"
                            disabled={isSearching || !query.trim()}
                            className="btn-primary h-9 px-4 text-[13px]"
                        >
                            {isSearching ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <span>Search</span>
                            )}
                        </button>
                    </div>
                </div>
            </form>

            {/* Toolbar row: Filters + Model selector */}
            {!agentMode && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-body dark:text-[#888] px-3 py-1.5 rounded-v-sm border border-transparent hover:border-hairline dark:hover:border-[rgba(255,255,255,0.1)] hover:bg-canvas dark:hover:bg-[rgba(255,255,255,0.04)] transition"
                    >
                        <Filter className="w-3.5 h-3.5" />
                        Filters
                        <ChevronDown className={`w-3.5 h-3.5 transition ${showFilters ? 'rotate-180' : ''}`} />
                    </button>

                    {/* Model selector */}
                    <div className="relative" ref={modelPickerRef}>
                        <button
                            type="button"
                            onClick={() => setShowModelPicker(!showModelPicker)}
                            className="inline-flex items-center gap-1.5 text-xs font-medium text-body dark:text-[#888] px-3 py-1.5 rounded-v-sm border border-transparent hover:border-hairline dark:hover:border-[rgba(255,255,255,0.1)] hover:bg-canvas dark:hover:bg-[rgba(255,255,255,0.04)] transition"
                        >
                            {selectedProvider === 'local' ? (
                                <Cpu className="w-3.5 h-3.5" />
                            ) : (
                                <Cloud className="w-3.5 h-3.5" />
                            )}
                            {currentModelLabel}
                            <ChevronDown className={`w-3.5 h-3.5 transition ${showModelPicker ? 'rotate-180' : ''}`} />
                        </button>

                        {showModelPicker && (
                            <div className="absolute left-0 top-full mt-1 z-50 w-72 bg-canvas dark:bg-[#171717] border border-hairline dark:border-[rgba(255,255,255,0.1)] rounded-v-md shadow-v-5 dark:shadow-v-dark-5 p-1 animate-slide-up">
                                {/* Cloud providers */}
                                <div className="px-2 pt-1.5 pb-1">
                                    <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">Cloud</div>
                                </div>
                                {CLOUD_PROVIDERS.map((p) => {
                                    const hasKey = modelConfig?.[p.keyField];
                                    return (
                                        <button
                                            key={p.id}
                                            onClick={() => selectModel(p.id)}
                                            disabled={!hasKey}
                                            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-v-sm text-sm transition ${
                                                selectedProvider === p.id
                                                    ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink font-medium'
                                                    : hasKey
                                                        ? 'text-ink dark:text-[#ededed] hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]'
                                                        : 'text-mute dark:text-[#444] cursor-not-allowed'
                                            }`}
                                        >
                                            <Cloud className="w-3.5 h-3.5 flex-shrink-0" />
                                            <span className="flex-1 text-left">{p.label}</span>
                                            {!hasKey && (
                                                <span className="text-[10px] font-mono text-mute">no key</span>
                                            )}
                                        </button>
                                    );
                                })}

                                {/* External */}
                                <div className="px-2 pt-2.5 pb-1">
                                    <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">External</div>
                                </div>
                                {['ollama', 'lmstudio'].map((id) => (
                                    <button
                                        key={id}
                                        onClick={() => selectModel(id)}
                                        className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-v-sm text-sm transition ${
                                            selectedProvider === id
                                                ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink font-medium'
                                                : 'text-ink dark:text-[#ededed] hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]'
                                        }`}
                                    >
                                        <Zap className="w-3.5 h-3.5 flex-shrink-0" />
                                        <span className="flex-1 text-left">{id === 'ollama' ? 'Ollama' : 'LM Studio'}</span>
                                    </button>
                                ))}

                                {/* Local models */}
                                {localModels.length > 0 && (
                                    <>
                                        <div className="px-2 pt-2.5 pb-1">
                                            <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">Local GGUF</div>
                                        </div>
                                        {localModels.map((m, i) => {
                                            const isActive = selectedProvider === 'local' && norm(m.path) === norm(selectedModelPath);
                                            return (
                                                <button
                                                    key={i}
                                                    onClick={() => selectModel('local', m.path)}
                                                    className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-v-sm text-sm transition ${
                                                        isActive
                                                            ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink font-medium'
                                                            : 'text-ink dark:text-[#ededed] hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]'
                                                    }`}
                                                >
                                                    <Cpu className="w-3.5 h-3.5 flex-shrink-0" />
                                                    <span className="flex-1 text-left truncate">{m.name || m.filename}</span>
                                                </button>
                                            );
                                        })}
                                    </>
                                )}
                            </div>
                        )}
                    </div>

                    {activeModel && (
                        <span className="chip">
                            <Sparkles className="w-3 h-3" />
                            {activeModel}
                        </span>
                    )}
                </div>
            )}

            {showFilters && !agentMode && (
                <div className="mt-3 card p-4 animate-slide-up">
                    <div className="grid sm:grid-cols-2 gap-4">
                        <div>
                            <div className="label">File type</div>
                            <div className="flex flex-wrap gap-1.5">
                                {FILE_TYPES.map((t) => {
                                    const active = filters.file_types.includes(t);
                                    return (
                                        <button
                                            key={t}
                                            type="button"
                                            onClick={() => toggleFileType(t)}
                                            className={`px-2.5 py-1 rounded-pill text-xs font-medium uppercase transition ${
                                                active
                                                    ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink'
                                                    : 'bg-canvas-soft dark:bg-[rgba(255,255,255,0.06)] text-body dark:text-[#888] border border-hairline dark:border-[rgba(255,255,255,0.1)] hover:border-hairline-strong dark:hover:border-[rgba(255,255,255,0.2)]'
                                            }`}
                                        >
                                            {t}
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                        <div>
                            <div className="label">Sort by</div>
                            <select
                                className="input"
                                value={filters.sort_by}
                                onChange={(e) => setFilters((f) => ({ ...f, sort_by: e.target.value }))}
                            >
                                {SORT_OPTIONS.map((o) => (
                                    <option key={o.value} value={o.value}>{o.label}</option>
                                ))}
                            </select>
                        </div>
                    </div>
                </div>
            )}

            {/* Error */}
            {error && !isSearching && (
                <div className="mt-6 p-4 bg-error-soft dark:bg-[rgba(238,0,0,0.1)] border border-error/20 rounded-v-md text-sm text-error-deep dark:text-[#ff6666]">
                    {error}
                </div>
            )}

            {/* Results */}
            {hasSearched && !agentMode && (
                <div className="mt-8">
                    {/* AI Synthesis */}
                    {(aiAnswer || isStreaming) && (
                        <section className="card-elevated p-5 mb-6 animate-slide-up">
                            <div className="flex items-center gap-2.5 mb-3">
                                <div className="w-7 h-7 rounded-v-sm bg-gradient-to-br from-gradient-develop-start to-gradient-develop-end text-white flex items-center justify-center">
                                    <Sparkles className="w-3.5 h-3.5" />
                                </div>
                                <div className="font-semibold text-sm text-ink dark:text-[#ededed] tracking-[-0.28px]">AI synthesis</div>
                                {isStreaming && (
                                    <div className="flex items-center gap-0.5 ml-1">
                                        <span className="typing-dot w-1.5 h-1.5 bg-ink dark:bg-[#ededed] rounded-full inline-block" />
                                        <span className="typing-dot w-1.5 h-1.5 bg-ink dark:bg-[#ededed] rounded-full inline-block" />
                                        <span className="typing-dot w-1.5 h-1.5 bg-ink dark:bg-[#ededed] rounded-full inline-block" />
                                    </div>
                                )}
                            </div>
                            <div className="prose-stream text-sm text-body dark:text-[#a1a1a1] leading-relaxed whitespace-pre-wrap">
                                {aiAnswer || (isStreaming ? 'Thinking...' : '')}
                            </div>
                        </section>
                    )}

                    {/* Results list */}
                    {isSearching ? (
                        <div className="space-y-3">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="card p-4">
                                    <div className="h-4 bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.04)] rounded w-1/3 mb-3 shimmer" />
                                    <div className="h-3 bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.04)] rounded w-full mb-1.5 shimmer" />
                                    <div className="h-3 bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.04)] rounded w-5/6 shimmer" />
                                </div>
                            ))}
                        </div>
                    ) : results.length > 0 ? (
                        <>
                            <div className="flex items-center justify-between mb-3">
                                <span className="font-mono text-[11px] uppercase tracking-[0.05em] text-mute">{results.length} {results.length === 1 ? 'result' : 'results'}</span>
                            </div>
                            <div className="space-y-3">
                                {results.map((r, i) => (
                                    <ResultCard key={r.faiss_idx ?? i} result={r} />
                                ))}
                            </div>
                        </>
                    ) : !error && (
                        <div className="text-center py-16">
                            <Search className="w-10 h-10 mx-auto mb-3 text-hairline dark:text-[rgba(255,255,255,0.1)]" />
                            <p className="text-sm font-medium text-body dark:text-[#888]">No results found</p>
                            <p className="text-xs mt-1 text-mute">Try a broader query or different keywords.</p>
                        </div>
                    )}
                </div>
            )}

            {hasSearched && agentMode && (
                <AgentView query={agentQuery} />
            )}
        </div>
    );
}
