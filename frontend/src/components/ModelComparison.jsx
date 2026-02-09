import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { GitCompare, Send, Loader2, AlertCircle } from 'lucide-react';

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
            // Use the summarize endpoint with the specific model
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
            // Fallback: try search endpoint
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
            <div className="text-center py-8 text-muted-foreground">
                <GitCompare className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Need at least 2 downloaded models</p>
                <p className="text-xs">Download more models to compare their outputs side-by-side</p>
            </div>
        );
    }

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center gap-2">
                <GitCompare className="w-5 h-5 text-primary" />
                <h3 className="font-semibold">Side-by-Side Comparison</h3>
            </div>

            {/* Model Selection */}
            <div className="grid grid-cols-2 gap-2">
                <select
                    value={model1}
                    onChange={(e) => setModel1(e.target.value)}
                    className="p-2 text-sm rounded-lg border border-border bg-background"
                >
                    {localModels.map((m) => (
                        <option key={m.path} value={m.path}>{m.name}</option>
                    ))}
                </select>
                <select
                    value={model2}
                    onChange={(e) => setModel2(e.target.value)}
                    className="p-2 text-sm rounded-lg border border-border bg-background"
                >
                    {localModels.map((m) => (
                        <option key={m.path} value={m.path}>{m.name}</option>
                    ))}
                </select>
            </div>

            {/* Query Input */}
            <div className="flex gap-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter a question to compare responses..."
                    className="flex-1 p-2 text-sm rounded-lg border border-border bg-background"
                    onKeyDown={(e) => e.key === 'Enter' && handleCompare()}
                />
                <button
                    onClick={handleCompare}
                    disabled={!query.trim() || loading.model1 || loading.model2}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                    {loading.model1 || loading.model2 ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                        <Send className="w-4 h-4" />
                    )}
                </button>
            </div>

            {error && (
                <div className="flex items-center gap-2 p-2 rounded-lg bg-destructive/10 text-destructive text-sm">
                    <AlertCircle className="w-4 h-4" />
                    {error}
                </div>
            )}

            {/* Results */}
            <div className="grid grid-cols-2 gap-2">
                {/* Model 1 Response */}
                <div className="p-3 rounded-lg border border-border bg-card min-h-[120px]">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium text-primary truncate">
                            {getModelName(model1)}
                        </span>
                        {results.model1?.time && (
                            <span className="text-xs text-muted-foreground">
                                {(results.model1.time / 1000).toFixed(1)}s
                            </span>
                        )}
                    </div>
                    {loading.model1 ? (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Generating...
                        </div>
                    ) : results.model1 ? (
                        <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                            {results.model1.text}
                        </p>
                    ) : (
                        <p className="text-xs text-muted-foreground italic">
                            Response will appear here
                        </p>
                    )}
                </div>

                {/* Model 2 Response */}
                <div className="p-3 rounded-lg border border-border bg-card min-h-[120px]">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-medium text-primary truncate">
                            {getModelName(model2)}
                        </span>
                        {results.model2?.time && (
                            <span className="text-xs text-muted-foreground">
                                {(results.model2.time / 1000).toFixed(1)}s
                            </span>
                        )}
                    </div>
                    {loading.model2 ? (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Generating...
                        </div>
                    ) : results.model2 ? (
                        <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                            {results.model2.text}
                        </p>
                    ) : (
                        <p className="text-xs text-muted-foreground italic">
                            Response will appear here
                        </p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ModelComparison;
