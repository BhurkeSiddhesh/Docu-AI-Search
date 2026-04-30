import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ModelComparison = () => {
    const [localModels, setLocalModels] = useState([]);
    const [model1, setModel1] = useState('');
    const [model2, setModel2] = useState('');
    const [query, setQuery] = useState('');
    const [results, setResults] = useState({ model1: null, model2: null });
    const [loading, setLoading] = useState({ model1: false, model2: false });
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchModels();
    }, []);

    const fetchModels = async () => {
        try {
            const response = await axios.get('http://localhost:8000/api/models/local');
            setLocalModels(response.data || []);
            if (response.data?.length >= 2) {
                setModel1(response.data[0].path);
                setModel2(response.data[1].path);
            } else if (response.data?.length === 1) {
                setModel1(response.data[0].path);
            }
        } catch (err) {
            console.error('Failed to fetch models:', err);
        }
    };

    const generateResponse = async (modelPath, modelKey) => {
        if (!modelPath || !query.trim()) return;

        setLoading(prev => ({ ...prev, [modelKey]: true }));
        setError(null);

        try {
            const startTime = Date.now();
            const response = await axios.post('http://localhost:8000/api/compare', {
                query: query,
                model_path: modelPath
            });
            const endTime = Date.now();

            setResults(prev => ({
                ...prev,
                [modelKey]: {
                    text: response.data.response || 'No response generated',
                    time: endTime - startTime
                }
            }));
        } catch (err) {
            try {
                const startTime = Date.now();
                const response = await axios.post('http://localhost:8000/api/search', {
                    query: query
                });
                const endTime = Date.now();

                setResults(prev => ({
                    ...prev,
                    [modelKey]: {
                        text: response.data.ai_answer || response.data.results?.[0]?.summary || 'No response',
                        time: endTime - startTime
                    }
                }));
            } catch (fallbackErr) {
                setError(`Failed to get response: ${err.message}`);
                setResults(prev => ({ ...prev, [modelKey]: null }));
            }
        } finally {
            setLoading(prev => ({ ...prev, [modelKey]: false }));
        }
    };

    const handleCompare = () => {
        setResults({ model1: null, model2: null });
        generateResponse(model1, 'model1');
        generateResponse(model2, 'model2');
    };

    const getModelName = (path) => {
        const model = localModels.find(m => m.path === path);
        return model?.name || path.split(/[/\\]/).pop()?.replace('.gguf', '') || 'Unknown';
    };

    if (localModels.length < 2) {
        return (
            <div className="p-24 rounded-[3rem] bg-[#f3f3fd] dark:bg-slate-950/40 border-2 border-dashed border-[#d1d1f0] dark:border-slate-800 text-center space-y-6">
                <div className="w-24 h-24 rounded-[2rem] bg-white dark:bg-slate-900 shadow-xl mx-auto flex items-center justify-center text-[#d1d1f0]">
                    <span className="material-symbols-outlined text-5xl">compare_arrows</span>
                </div>
                <div>
                    <p className="text-xl font-bold font-headline text-slate-400">Comparison Engine Offline</p>
                    <p className="text-sm opacity-60 font-medium">Download at least 2 models to enable side-by-side behavioral analysis.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-10 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                    <span className="material-symbols-outlined">compare_arrows</span>
                </div>
                <div>
                    <h3 className="text-2xl font-bold font-headline">Neural Comparison</h3>
                    <p className="text-xs opacity-60 font-medium mt-0.5 uppercase tracking-widest font-black">Analyze behavioral variance between models</p>
                </div>
            </div>

            {/* Selector and Input Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 bg-white dark:bg-slate-900 p-8 rounded-[2.5rem] border border-[#f3f3fd] dark:border-slate-800 shadow-sm">
                <div className="space-y-4">
                    <p className="text-[10px] font-black uppercase tracking-widest opacity-40 ml-4">Select neural agents</p>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="relative">
                            <select
                                value={model1}
                                onChange={(e) => setModel1(e.target.value)}
                                className="w-full pl-5 pr-10 py-4 bg-[#f3f3fd] dark:bg-slate-950 rounded-2xl text-sm font-bold appearance-none focus:ring-2 focus:ring-primary/20 border-none"
                            >
                                {localModels.map((m) => (
                                    <option key={m.path} value={m.path}>{m.name}</option>
                                ))}
                            </select>
                            <span className="material-symbols-outlined absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none opacity-40">expand_more</span>
                        </div>
                        <div className="relative">
                            <select
                                value={model2}
                                onChange={(e) => setModel2(e.target.value)}
                                className="w-full pl-5 pr-10 py-4 bg-[#f3f3fd] dark:bg-slate-950 rounded-2xl text-sm font-bold appearance-none focus:ring-2 focus:ring-primary/20 border-none"
                            >
                                {localModels.map((m) => (
                                    <option key={m.path} value={m.path}>{m.name}</option>
                                ))}
                            </select>
                            <span className="material-symbols-outlined absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none opacity-40">expand_more</span>
                        </div>
                    </div>
                </div>

                <div className="space-y-4">
                    <p className="text-[10px] font-black uppercase tracking-widest opacity-40 ml-4">Initialize Query</p>
                    <div className="flex gap-3">
                        <input
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="Enter test prompt..."
                            className="flex-1 px-6 py-4 bg-[#f3f3fd] dark:bg-slate-950 rounded-2xl text-sm font-bold placeholder:opacity-30 focus:ring-2 focus:ring-primary/20 border-none"
                            onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
                        />
                        <button
                            onClick={handleCompare}
                            disabled={!query.trim() || loading.model1 || loading.model2}
                            className="w-14 h-14 bg-primary text-white rounded-2xl flex items-center justify-center hover:shadow-xl hover:shadow-primary/20 transition-all active:scale-95 disabled:opacity-50"
                        >
                            {loading.model1 || loading.model2 ? (
                                <span className="material-symbols-outlined animate-spin">progress_activity</span>
                            ) : (
                                <span className="material-symbols-outlined">send</span>
                            )}
                        </button>
                    </div>
                </div>
            </div>

            {error && (
                <div className="p-5 rounded-3xl bg-red-500/10 text-red-500 text-sm font-bold border border-red-500/20 flex items-center gap-3">
                    <span className="material-symbols-outlined">error</span>
                    {error}
                </div>
            )}

            {/* Results Display */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Model 1 Response */}
                <div className="group space-y-4">
                    <div className="flex items-center justify-between px-6">
                        <div className="flex items-center gap-3">
                            <div className="w-2 h-2 rounded-full bg-amber-500"></div>
                            <span className="text-[10px] font-black uppercase tracking-widest opacity-40 truncate max-w-[200px]">
                                {getModelName(model1)}
                            </span>
                        </div>
                        {results.model1?.time && (
                            <div className="flex items-center gap-2 text-[10px] font-black text-amber-500">
                                <span className="material-symbols-outlined text-xs">timer</span>
                                {(results.model1.time / 1000).toFixed(1)}S
                            </div>
                        )}
                    </div>
                    
                    <div className="p-8 rounded-[2.5rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm min-h-[300px] transition-all group-hover:border-amber-500/30">
                        {loading.model1 ? (
                            <div className="h-full flex flex-col items-center justify-center gap-4 text-amber-500/40">
                                <span className="material-symbols-outlined text-4xl animate-pulse">neurology</span>
                                <p className="text-[10px] font-black uppercase tracking-widest">Generating Output...</p>
                            </div>
                        ) : results.model1 ? (
                            <p className="text-sm leading-relaxed font-medium opacity-80 whitespace-pre-wrap">
                                {results.model1.text}
                            </p>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center opacity-10">
                                <span className="material-symbols-outlined text-6xl">cloud_done</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Model 2 Response */}
                <div className="group space-y-4">
                    <div className="flex items-center justify-between px-6">
                        <div className="flex items-center gap-3">
                            <div className="w-2 h-2 rounded-full bg-primary"></div>
                            <span className="text-[10px] font-black uppercase tracking-widest opacity-40 truncate max-w-[200px]">
                                {getModelName(model2)}
                            </span>
                        </div>
                        {results.model2?.time && (
                            <div className="flex items-center gap-2 text-[10px] font-black text-primary">
                                <span className="material-symbols-outlined text-xs">timer</span>
                                {(results.model2.time / 1000).toFixed(1)}S
                            </div>
                        )}
                    </div>
                    
                    <div className="p-8 rounded-[2.5rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm min-h-[300px] transition-all group-hover:border-primary/30">
                        {loading.model2 ? (
                            <div className="h-full flex flex-col items-center justify-center gap-4 text-primary/40">
                                <span className="material-symbols-outlined text-4xl animate-pulse">neurology</span>
                                <p className="text-[10px] font-black uppercase tracking-widest">Generating Output...</p>
                            </div>
                        ) : results.model2 ? (
                            <p className="text-sm leading-relaxed font-medium opacity-80 whitespace-pre-wrap">
                                {results.model2.text}
                            </p>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center opacity-10">
                                <span className="material-symbols-outlined text-6xl">cloud_done</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelComparison;
