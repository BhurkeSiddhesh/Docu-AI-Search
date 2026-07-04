import React, { useEffect, useRef, useState } from 'react';
import { Search, Sparkles, Loader2, Bot, ChevronDown, Filter } from 'lucide-react';
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

    const inputRef = useRef(null);
    const streamAbortRef = useRef(null);
    const toast = useToast();

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
                <div className="text-center mb-10 mt-2 animate-fade-in">
                    <svg viewBox="0 0 520 200" className="w-full max-w-sm mx-auto mb-4" aria-hidden="true">
                        <defs>
                            <linearGradient id="hero-doc-a" x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stopColor="#6366f1" />
                                <stop offset="100%" stopColor="#8b5cf6" />
                            </linearGradient>
                            <linearGradient id="hero-lens" x1="0" y1="0" x2="1" y2="1">
                                <stop offset="0%" stopColor="#818cf8" />
                                <stop offset="100%" stopColor="#6366f1" />
                            </linearGradient>
                            <radialGradient id="hero-glow" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stopColor="#6366f1" stopOpacity="0.16" />
                                <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
                            </radialGradient>
                        </defs>
                        <ellipse cx="260" cy="105" rx="190" ry="85" fill="url(#hero-glow)" />

                        {/* orbiting topic dots */}
                        <circle cx="110" cy="52" r="5" fill="#f43f5e" opacity="0.8" />
                        <circle cx="408" cy="44" r="6" fill="#10b981" opacity="0.8" />
                        <circle cx="92" cy="150" r="4" fill="#f97316" opacity="0.8" />
                        <circle cx="424" cy="148" r="5" fill="#3b82f6" opacity="0.8" />
                        <g stroke="#94a3b8" strokeOpacity="0.4" strokeWidth="1.2" strokeDasharray="2 4">
                            <line x1="118" y1="57" x2="200" y2="86" />
                            <line x1="400" y1="50" x2="322" y2="80" />
                            <line x1="100" y1="146" x2="196" y2="124" />
                            <line x1="416" y1="144" x2="324" y2="122" />
                        </g>

                        {/* stacked documents */}
                        <g transform="translate(196 56) rotate(-6 0 0)">
                            <rect width="74" height="94" rx="10" fill="#e2e8f0" className="dark:opacity-20" />
                        </g>
                        <g transform="translate(252 52) rotate(5 0 0)">
                            <rect width="74" height="94" rx="10" fill="#cbd5e1" className="dark:opacity-25" />
                        </g>
                        <g transform="translate(222 48)">
                            <rect width="78" height="100" rx="10" fill="url(#hero-doc-a)" />
                            <rect x="14" y="20" width="50" height="6" rx="3" fill="white" opacity="0.9" />
                            <rect x="14" y="34" width="38" height="5" rx="2.5" fill="white" opacity="0.6" />
                            <rect x="14" y="46" width="44" height="5" rx="2.5" fill="white" opacity="0.6" />
                            <rect x="14" y="58" width="30" height="5" rx="2.5" fill="white" opacity="0.4" />
                        </g>

                        {/* magnifying lens */}
                        <g transform="translate(296 108)">
                            <circle r="30" fill="white" fillOpacity="0.65" stroke="url(#hero-lens)" strokeWidth="7" />
                            <line x1="22" y1="22" x2="44" y2="44" stroke="url(#hero-lens)" strokeWidth="9" strokeLinecap="round" />
                            <path d="M -12 2 q 6 -12 20 -8" stroke="#6366f1" strokeWidth="3" fill="none" strokeLinecap="round" opacity="0.6" />
                        </g>
                    </svg>
                    <h1 className="text-3xl sm:text-4xl font-bold text-slate-900 dark:text-slate-50 mb-3">
                        Search your documents
                    </h1>
                    <p className="text-slate-600 dark:text-slate-400 text-base">
                        Ask natural language questions across your indexed files.
                    </p>
                </div>
            )}

            {/* Search bar */}
            <form onSubmit={onSubmit}>
                <div className="relative">
                    <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400">
                        <Search className="w-5 h-5" />
                    </div>
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder={agentMode ? 'Ask the research agent…' : 'Search your documents… (Ctrl+K)'}
                        disabled={isSearching}
                        className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl pl-12 pr-32 py-4 text-base text-slate-900 dark:text-slate-50 placeholder:text-slate-400 focus:border-primary focus:ring-4 focus:ring-primary/10 outline-none transition"
                        aria-label="Search"
                    />
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1.5">
                        <button
                            type="button"
                            onClick={() => setAgentMode(!agentMode)}
                            title={agentMode ? 'Agent mode on' : 'Enable agent mode'}
                            className={`p-2 rounded-lg transition ${
                                agentMode
                                    ? 'bg-primary/10 text-primary'
                                    : 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800'
                            }`}
                        >
                            <Bot className="w-4 h-4" />
                        </button>
                        <button
                            type="submit"
                            disabled={isSearching || !query.trim()}
                            className="inline-flex items-center gap-1.5 bg-primary text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary/90 disabled:opacity-40 disabled:cursor-not-allowed transition"
                        >
                            {isSearching ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : (
                                <>
                                    <span>Search</span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            </form>

            {/* Filter row */}
            {!agentMode && (
                <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                        onClick={() => setShowFilters(!showFilters)}
                        className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-400 px-3 py-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                    >
                        <Filter className="w-3.5 h-3.5" />
                        Filters
                        <ChevronDown className={`w-3.5 h-3.5 transition ${showFilters ? 'rotate-180' : ''}`} />
                    </button>

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
                                            className={`px-2.5 py-1 rounded-md text-xs font-medium uppercase transition ${
                                                active
                                                    ? 'bg-primary text-white'
                                                    : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
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
                <div className="mt-6 p-4 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 rounded-lg text-sm text-red-900 dark:text-red-200">
                    {error}
                </div>
            )}

            {/* Results */}
            {hasSearched && !agentMode && (
                <div className="mt-8">
                    {/* AI Synthesis */}
                    {(aiAnswer || isStreaming) && (
                        <section className="card p-5 mb-6 animate-slide-up">
                            <div className="flex items-center gap-2.5 mb-3">
                                <div className="w-7 h-7 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
                                    <Sparkles className="w-4 h-4" />
                                </div>
                                <div className="font-semibold text-sm text-slate-900 dark:text-slate-50">AI synthesis</div>
                                {isStreaming && (
                                    <div className="flex items-center gap-0.5 ml-1">
                                        <span className="typing-dot w-1.5 h-1.5 bg-primary rounded-full inline-block" />
                                        <span className="typing-dot w-1.5 h-1.5 bg-primary rounded-full inline-block" />
                                        <span className="typing-dot w-1.5 h-1.5 bg-primary rounded-full inline-block" />
                                    </div>
                                )}
                            </div>
                            <div className="prose-stream text-sm text-slate-700 dark:text-slate-300 leading-relaxed whitespace-pre-wrap">
                                {aiAnswer || (isStreaming ? 'Thinking…' : '')}
                            </div>
                        </section>
                    )}

                    {/* Results list */}
                    {isSearching ? (
                        <div className="space-y-3">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="card p-4">
                                    <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded w-1/3 mb-3" />
                                    <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded w-full mb-1.5" />
                                    <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded w-5/6" />
                                </div>
                            ))}
                        </div>
                    ) : results.length > 0 ? (
                        <>
                            <div className="flex items-center justify-between mb-3 text-xs font-medium text-slate-500 dark:text-slate-400">
                                <span>{results.length} {results.length === 1 ? 'result' : 'results'}</span>
                            </div>
                            <div className="space-y-3">
                                {results.map((r, i) => (
                                    <ResultCard key={r.faiss_idx ?? i} result={r} />
                                ))}
                            </div>
                        </>
                    ) : !error && (
                        <div className="text-center py-16 text-slate-500 dark:text-slate-400">
                            <Search className="w-10 h-10 mx-auto mb-3 opacity-30" />
                            <p className="text-sm font-medium">No results found</p>
                            <p className="text-xs mt-1">Try a broader query or different keywords.</p>
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
