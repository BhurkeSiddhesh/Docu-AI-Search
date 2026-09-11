import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Search, Sparkles, Loader2, Bot, ChevronDown, Cpu, Cloud, Zap, FileText, FileSpreadsheet, Presentation, SlidersHorizontal } from 'lucide-react';
import { motion } from 'framer-motion';
import ResultCard from './ResultCard';
import AgentView from './AgentView';
import ErrorBoundary from './ErrorBoundary';
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

function EmptySearchArt({ focused }) {
    const flowDuration = focused ? 3.2 : 4.4;
    const sourceFiles = [
        { Icon: FileText, tone: 'text-rose-500', delay: 0 },
        { Icon: FileSpreadsheet, tone: 'text-emerald-500', delay: 0.16 },
        { Icon: Presentation, tone: 'text-orange-500', delay: 0.32 },
    ];
    const resultLines = [
        { width: '100%', delay: 1.95 },
        { width: '86%', delay: 2.08 },
        { width: '68%', delay: 2.21 },
    ];

    return (
        <div
            className="semantic-lens-visual relative mx-auto mb-10 w-full max-w-[600px] h-[220px] overflow-hidden"
            aria-hidden="true"
            data-testid="semantic-lens-visual"
        >
            <div className="absolute left-[124px] right-[152px] top-[103px] h-px bg-[rgba(23,23,23,0.14)] dark:bg-[rgba(255,255,255,0.16)]" />
            <div className="absolute left-[214px] top-[92px] h-[22px] w-px bg-[rgba(23,23,23,0.08)] dark:bg-[rgba(255,255,255,0.12)]" />
            <div className="absolute right-[172px] top-[92px] h-[22px] w-px bg-[rgba(23,23,23,0.08)] dark:bg-[rgba(255,255,255,0.12)]" />

            <div className="absolute left-[18px] top-[38px] flex flex-col gap-3">
                {sourceFiles.map(({ Icon, tone, delay }, index) => (
                    <motion.div
                        key={tone}
                        className="relative h-12 w-[108px] rounded-[14px] bg-white/94 dark:bg-[#161616]/94 border border-[rgba(0,0,0,0.08)] dark:border-[rgba(255,255,255,0.1)] shadow-[0_12px_36px_-30px_rgba(0,0,0,0.65),inset_0_1px_0_rgba(255,255,255,0.76)] dark:shadow-[0_14px_44px_-32px_rgba(0,0,0,1),inset_0_1px_0_rgba(255,255,255,0.08)]"
                        animate={{ x: focused ? [0, 4, 0] : [0, 2, 0] }}
                        transition={{ duration: 2.2, repeat: Infinity, delay, ease: 'easeInOut' }}
                    >
                        <Icon className={`absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 ${tone}`} strokeWidth={1.8} />
                        <span className="absolute left-10 top-[15px] h-1.5 w-12 rounded-full bg-[rgba(23,23,23,0.16)] dark:bg-[rgba(255,255,255,0.16)]" />
                        <span className="absolute left-10 top-[25px] h-1.5 w-8 rounded-full bg-[rgba(23,23,23,0.1)] dark:bg-[rgba(255,255,255,0.1)]" />
                        <motion.span
                            className="absolute left-[112px] top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-link dark:bg-[#3291ff]"
                            animate={{ x: [0, 112, 112], opacity: [0, 1, 0], scale: [0.8, 1, 0.7] }}
                            transition={{ duration: flowDuration, repeat: Infinity, delay: index * 0.22, times: [0, 0.5, 0.68], ease: 'easeInOut' }}
                        />
                    </motion.div>
                ))}
            </div>

            <motion.div
                className="absolute left-1/2 top-[54px] h-[104px] w-[104px] -translate-x-1/2 rounded-[30px] border border-[rgba(23,23,23,0.12)] dark:border-[rgba(255,255,255,0.16)] bg-white dark:bg-[#111] shadow-[0_18px_58px_-38px_rgba(0,0,0,0.9),inset_0_1px_0_rgba(255,255,255,0.72)] dark:shadow-[0_20px_68px_-40px_rgba(0,0,0,1),inset_0_1px_0_rgba(255,255,255,0.09)]"
                animate={{
                    scale: focused ? [1, 1.05, 1, 1.03, 1] : [1, 1.025, 1, 1.012, 1],
                    borderColor: focused
                        ? ['rgba(23,23,23,0.12)', 'rgba(0,112,243,0.46)', 'rgba(23,23,23,0.12)', 'rgba(0,112,243,0.34)', 'rgba(23,23,23,0.12)']
                        : ['rgba(23,23,23,0.12)', 'rgba(0,112,243,0.26)', 'rgba(23,23,23,0.12)', 'rgba(0,112,243,0.2)', 'rgba(23,23,23,0.12)'],
                }}
                transition={{ duration: flowDuration, repeat: Infinity, ease: 'easeInOut' }}
            >
                <div className="absolute inset-0 flex items-center justify-center">
                    <Search className="w-9 h-9 text-ink dark:text-[#ededed]" strokeWidth={1.75} />
                </div>
                <motion.div
                    className="absolute -right-[48px] top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-link dark:bg-[#3291ff]"
                    animate={{ x: [0, 82, 82], opacity: [0, 1, 0], scale: [0.8, 1.1, 0.75] }}
                    transition={{ duration: flowDuration, repeat: Infinity, delay: 1.34, times: [0, 0.48, 0.68], ease: 'easeInOut' }}
                />
            </motion.div>

            <motion.div
                className="absolute right-[18px] top-[48px] h-[124px] w-[166px] rounded-[22px] bg-white/94 dark:bg-[#161616]/94 border border-[rgba(0,0,0,0.08)] dark:border-[rgba(255,255,255,0.1)] shadow-[0_18px_58px_-40px_rgba(0,0,0,0.78),inset_0_1px_0_rgba(255,255,255,0.72)] dark:shadow-[0_20px_68px_-42px_rgba(0,0,0,1),inset_0_1px_0_rgba(255,255,255,0.08)]"
                animate={{
                    y: focused ? [0, -3, 0] : [0, 2, 0],
                    opacity: [0.82, 0.82, 1, 1, 0.82],
                    scale: [0.99, 0.99, 1, 1, 0.99],
                }}
                transition={{
                    duration: flowDuration,
                    repeat: Infinity,
                    times: [0, 0.46, 0.62, 0.84, 1],
                    ease: 'easeInOut',
                }}
            >
                <div className="absolute left-4 top-4 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-link dark:text-[#3291ff]" />
                    <span className="h-1.5 w-16 rounded-full bg-[rgba(23,23,23,0.18)] dark:bg-[rgba(255,255,255,0.18)]" />
                </div>
                <div className="absolute left-4 right-4 top-12 space-y-2.5">
                    {resultLines.map((line) => (
                        <motion.span
                            key={line.width}
                            className="block h-1.5 rounded-full bg-[rgba(23,23,23,0.16)] dark:bg-[rgba(255,255,255,0.16)]"
                            style={{ maxWidth: line.width }}
                            animate={{ width: ['0%', line.width, line.width, '0%'], opacity: [0, 1, 1, 0] }}
                            transition={{ duration: flowDuration, repeat: Infinity, delay: line.delay, times: [0, 0.16, 0.68, 1], ease: 'easeInOut' }}
                        />
                    ))}
                </div>
                <motion.div
                    className="absolute bottom-4 left-4 flex items-center gap-1.5"
                    animate={{ opacity: [0, 0, 1, 1, 0] }}
                    transition={{ duration: flowDuration, repeat: Infinity, times: [0, 0.56, 0.68, 0.9, 1], ease: 'easeInOut' }}
                >
                    <span className="h-2 w-2 rounded-full bg-emerald-500" />
                    <span className="h-1.5 w-16 rounded-full bg-[rgba(23,23,23,0.12)] dark:bg-[rgba(255,255,255,0.12)]" />
                </motion.div>
            </motion.div>
        </div>
    );
}

function SearchRetrievalProgress({ progressLabel = 'Searching library' }) {
    const fileTypes = ['PDF', 'DOCX', 'CSV', 'MD', 'PPT'];
    return (
        <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 card-soft p-3"
            role="status"
            aria-label={progressLabel}
        >
            <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-sm font-medium text-ink dark:text-[#ededed]">
                    <Search className="w-4 h-4 search-process-pulse" />
                    {progressLabel}
                </div>
                <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">Retrieval</div>
            </div>
            <div className="mt-3 h-12 relative overflow-hidden rounded-v-sm bg-canvas dark:bg-[#0a0a0a] border border-hairline dark:border-[rgba(255,255,255,0.08)]">
                {fileTypes.map((type, index) => (
                    <motion.div
                        key={type}
                        className="absolute top-3 px-2 py-1 rounded-v-sm border border-hairline dark:border-[rgba(255,255,255,0.1)] bg-canvas-soft dark:bg-[rgba(255,255,255,0.04)] font-mono text-[10px] text-mute"
                        initial={{ x: `${8 + index * 18}%`, opacity: 0.35 }}
                        animate={{ x: ['10%', '45%', '78%'], opacity: [0.35, 1, 0.35] }}
                        transition={{ duration: 1.4, repeat: Infinity, delay: index * 0.1, ease: 'easeInOut' }}
                    >
                        {type}
                    </motion.div>
                ))}
            </div>
        </motion.div>
    );
}

export default function SearchView({ pendingQuery }) {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [aiAnswer, setAiAnswer] = useState('');
    const [activeModel, setActiveModel] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [error, setError] = useState(null);

    const [isInputFocused, setIsInputFocused] = useState(false);
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
            const [cfgRes, modelsRes] = await Promise.all([
                api.getConfig(),
                api.listLocalModels().catch(() => ({ data: [] })),
            ]);
            const cfg = cfgRes.data || {};
            setModelConfig(cfg);
            setLocalModels(modelsRes.data || []);
            setSelectedProvider(cfg.provider || 'openai');
            setSelectedModelPath(cfg.local_model_path || '');
        } catch {
            // silent - model picker will be hidden if no config
        }
    }, []);

    useEffect(() => {
        loadModelOptions();
    }, [loadModelOptions]);

    useEffect(() => {
        const handleConfigUpdated = () => {
            loadModelOptions();
        };
        window.addEventListener('docu-config-updated', handleConfigUpdated);
        return () => window.removeEventListener('docu-config-updated', handleConfigUpdated);
    }, [loadModelOptions]);

    // Close model picker on outside click
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (modelPickerRef.current && !modelPickerRef.current.contains(e.target)) {
                setShowModelPicker(false);
            }
        };
        if (showModelPicker) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [showModelPicker]);

    // Abort stream on unmount
    useEffect(() => {
        return () => streamAbortRef.current?.abort();
    }, []);

    // Respond to query from another tab (e.g. library search button)
    useEffect(() => {
        if (pendingQuery?.q) {
            setQuery(pendingQuery.q);
            performSearch(pendingQuery.q);
        }
    }, [pendingQuery]);

    // Global shortcut: Ctrl+K or Cmd+K to focus search input
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
            await loadModelOptions();
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

    const norm = (p) => (p || '').replace(/\\/g, '/').toLowerCase();

    const performSearch = async (searchQuery) => {
        if (!searchQuery.trim()) return;

        streamAbortRef.current?.abort();

        setError(null);
        setHasSearched(true);
        setResults([]);
        setAiAnswer('');

        if (agentMode) {
            setAgentQuery(searchQuery);
            return;
        }

        setIsSearching(true);
        try {
            const searchFilters = {
                query: searchQuery,
                file_types: filters.file_types.length ? filters.file_types : undefined,
                sort_by: filters.sort_by !== 'relevance' ? filters.sort_by : undefined,
                min_score: filters.min_score ?? undefined,
            };
            const res = await api.search(searchQuery, searchFilters);
            setResults(res.data.results || []);
            setActiveModel(res.data.active_model || '');
            setIsSearching(false);

            if ((res.data.results || []).length > 0) {
                const controller = new AbortController();
                streamAbortRef.current = controller;
                setIsStreaming(true);
                let acc = '';
                try {
                    const context = res.data.results.map((r) => r.summary || r.document).slice(0, 4);
                    await api.streamAnswer(searchQuery, context, (chunk) => {
                        if (controller.signal.aborted) return;
                        acc += chunk;
                        setAiAnswer(acc);
                    }, controller.signal);
                } catch (streamErr) {
                    if (streamErr?.name !== 'AbortError') {
                        console.error('Stream error:', streamErr);
                        toast.error('AI answer stream failed. Search results are still available.');
                    }
                } finally {
                    if (streamAbortRef.current === controller) {
                        setIsStreaming(false);
                    }
                }
            }
        } catch (err) {
            setIsSearching(false);
            const msg = err.response?.data?.detail || err.message || 'Search failed';
            setError(msg);
            if (err.response?.status === 400) {
                toast.error('No index yet. Configure and index a folder in Settings.');
            } else {
                toast.error(msg);
            }
        }
    };

    const handleSearch = (e) => {
        e.preventDefault();
        performSearch(query);
    };

    const toggleFileType = (type) => {
        setFilters((prev) => ({
            ...prev,
            file_types: prev.file_types.includes(type)
                ? prev.file_types.filter((t) => t !== type)
                : [...prev.file_types, type],
        }));
    };

    return (
        <div className={`h-full min-h-0 max-w-5xl w-full mx-auto px-4 sm:px-6 pt-6 lg:pt-10 pb-6 ${hasSearched ? 'flex flex-col' : 'overflow-y-auto'}`}>
            {/* Search hero & bar */}
            <div className="shrink-0">
                {!hasSearched && (
                    <div className="text-center mb-10 mt-1 lg:mt-2 animate-fade-in">
                        <EmptySearchArt focused={isInputFocused} />
                        <h1 className="text-[44px] sm:text-[56px] leading-[0.96] font-semibold tracking-[-0.045em] text-ink dark:text-[#ededed] mb-4">
                            Ask your library.
                        </h1>
                        <p className="text-body dark:text-[#888] text-[17px] sm:text-[19px] leading-7 max-w-xl mx-auto">
                            Search locally, surface the exact passage, and synthesize an answer without losing the source.
                        </p>
                    </div>
                )}

                {/* Input row */}
                <form onSubmit={handleSearch}>
                    <div className="relative group">
                        <div className="absolute left-5 top-1/2 -translate-y-1/2 text-mute group-focus-within:text-ink dark:group-focus-within:text-[#ededed] transition-colors">
                            <Search className="w-5 h-5" strokeWidth={1.8} />
                        </div>
                        <input
                            ref={inputRef}
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onFocus={() => setIsInputFocused(true)}
                            onBlur={() => setIsInputFocused(false)}
                            placeholder={agentMode ? 'Ask the research agent...' : 'Ask a question about your documents...'}
                            disabled={isSearching}
                            className="w-full bg-white/92 dark:bg-[#101010] border border-[rgba(0,0,0,0.08)] dark:border-[rgba(255,255,255,0.13)] rounded-[18px] pl-14 pr-36 h-16 text-[17px] text-ink dark:text-[#ededed] placeholder:text-mute outline-none transition-all shadow-[0_20px_60px_-45px_rgba(0,0,0,0.8),inset_0_1px_0_rgba(255,255,255,0.9)] dark:shadow-[0_22px_64px_-44px_rgba(0,0,0,1),inset_0_1px_0_rgba(255,255,255,0.06)] focus:border-[rgba(0,0,0,0.22)] dark:focus:border-[rgba(255,255,255,0.28)] focus:shadow-[0_24px_70px_-44px_rgba(0,0,0,0.9),0_0_0_4px_rgba(0,112,243,0.08),inset_0_1px_0_rgba(255,255,255,0.95)] dark:focus:shadow-[0_24px_70px_-44px_rgba(0,0,0,1),0_0_0_4px_rgba(50,145,255,0.12),inset_0_1px_0_rgba(255,255,255,0.08)]"
                            aria-label="Search"
                        />
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                            <button
                                type="button"
                                onClick={() => setAgentMode(!agentMode)}
                                title={agentMode ? 'Agent mode on' : 'Enable agent mode'}
                                className={`p-2.5 rounded-[12px] transition ${
                                    agentMode
                                        ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink'
                                        : 'text-mute hover:text-ink hover:bg-canvas-soft-2 dark:hover:text-[#ededed] dark:hover:bg-[rgba(255,255,255,0.06)]'
                                }`}
                            >
                                <Bot className="w-4 h-4" strokeWidth={1.8} />
                            </button>
                            <button
                                type="submit"
                                disabled={isSearching || !query.trim()}
                                className="btn-primary h-11 px-5 text-[14px]"
                            >
                                {isSearching ? <Loader2 className="w-4 h-4 animate-spin" /> : <span>Search</span>}
                            </button>
                        </div>
                    </div>
                </form>

                {isSearching && !agentMode && (
                    <SearchRetrievalProgress progressLabel="Searching library" />
                )}

                {/* Toolbar row: Filters + Model selector */}
                {!agentMode && (
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                        <button
                            onClick={() => setShowFilters(!showFilters)}
                            className="inline-flex items-center gap-1.5 text-xs font-medium text-body dark:text-[#888] px-3 py-1.5 rounded-v-sm border border-transparent hover:border-hairline dark:hover:border-[rgba(255,255,255,0.1)] hover:bg-canvas dark:hover:bg-[rgba(255,255,255,0.04)] transition"
                        >
                            <SlidersHorizontal className="w-3.5 h-3.5" />
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
                                <div
                                    data-testid="model-picker-dropdown"
                                    className="absolute left-0 bottom-full mb-1 z-50 w-72 max-h-[380px] overflow-y-auto bg-canvas dark:bg-[#171717] border border-hairline dark:border-[rgba(255,255,255,0.1)] rounded-v-md shadow-v-5 dark:shadow-v-dark-5 p-1 animate-slide-up"
                                >
                                    {/* Local models */}
                                    {localModels.length > 0 && (
                                        <>
                                            <div className="px-2 pt-1.5 pb-1">
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

                                    {/* Cloud providers */}
                                    <div className={`px-2 ${localModels.length > 0 ? 'pt-2.5' : 'pt-1.5'} pb-1`}>
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
                                    onChange={(e) => setFilters((prev) => ({ ...prev, sort_by: e.target.value }))}
                                >
                                    {SORT_OPTIONS.map((o) => (
                                        <option key={o.value} value={o.value}>{o.label}</option>
                                    ))}
                                </select>
                            </div>
                        </div>
                    </div>
                )}

                {error && !isSearching && (
                    <div className="mt-6 p-4 bg-error-soft dark:bg-[rgba(238,0,0,0.1)] border border-error/20 rounded-v-md text-sm text-error-deep dark:text-[#ff6666]">
                        {error}
                    </div>
                )}
            </div>

            {/* Results section */}
            {hasSearched && !agentMode && (
                <div className="mt-8 flex-1 min-h-0 overflow-y-auto -mx-1 px-1">
                    {/* AI Answer streaming box */}
                    {(aiAnswer || isStreaming) && (
                        <motion.section
                            initial={{ opacity: 0, y: 8 }}
                            animate={{
                                opacity: 1,
                                y: 0,
                                boxShadow: isStreaming
                                    ? ['0 2px 2px rgba(0,0,0,0.04), 0 8px 16px -4px rgba(0,0,0,0.04)', '0 0 0 1px rgba(0,112,243,0.18), 0 12px 32px -12px rgba(0,112,243,0.35)', '0 2px 2px rgba(0,0,0,0.04), 0 8px 16px -4px rgba(0,0,0,0.04)']
                                    : undefined,
                            }}
                            transition={{
                                opacity: { duration: 0.2 },
                                y: { duration: 0.2 },
                                boxShadow: { duration: 1.6, repeat: isStreaming ? Infinity : 0, ease: 'easeInOut' },
                            }}
                            className="card-elevated p-5 mb-6"
                        >
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
                            {/^(Error:|\n?\[ERROR\])/.test((aiAnswer || '').trim()) ? (
                                <div className="text-sm text-error leading-relaxed flex items-start gap-2">
                                    <span className="mt-0.5">⚠</span>
                                    <span>{aiAnswer.replace(/^\[ERROR\]\s*/, '').replace(/^Error:\s*/, '')} — configure a model under Settings → Local Models or Cloud Providers.</span>
                                </div>
                            ) : (
                                <div className="prose-stream text-sm text-body dark:text-[#a1a1a1] leading-relaxed whitespace-pre-wrap">
                                    {aiAnswer || (isStreaming ? 'Thinking...' : '')}
                                    {isStreaming && <span className="streaming-cursor" aria-hidden="true" />}
                                </div>
                            )}
                        </motion.section>
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
                                <span className="font-mono text-[11px] uppercase tracking-[0.05em] text-mute">
                                    {results.length} {results.length === 1 ? 'result' : 'results'}
                                </span>
                            </div>
                            <div className="space-y-3">
                                {results.map((r, i) => (
                                    <motion.div
                                        key={r.faiss_idx ?? i}
                                        initial={{ opacity: 0, y: 8 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ duration: 0.22, delay: Math.min(i * 0.04, 0.32), ease: 'easeOut' }}
                                    >
                                        <ResultCard result={r} />
                                    </motion.div>
                                ))}
                            </div>
                        </>
                    ) : !error ? (
                        <div className="text-center py-16">
                            <Search className="w-10 h-10 mx-auto mb-3 text-hairline dark:text-[rgba(255,255,255,0.1)]" />
                            <p className="text-sm font-medium text-body dark:text-[#888]">No results found</p>
                            <p className="text-xs mt-1 text-mute">Try a broader query or different keywords.</p>
                        </div>
                    ) : null}
                </div>
            )}

            {/* Agent Mode View */}
            {hasSearched && agentMode && (
                <div className="flex-1 min-h-0 overflow-y-auto -mx-1 px-1">
                    <ErrorBoundary>
                        <AgentView query={agentQuery} />
                    </ErrorBoundary>
                </div>
            )}
        </div>
    );
}
