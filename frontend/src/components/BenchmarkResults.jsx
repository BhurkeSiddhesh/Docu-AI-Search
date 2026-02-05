import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart3, Play, Loader2, Trophy, Zap, Brain, HardDrive } from 'lucide-react';

const BenchmarkResults = () => {
    const [results, setResults] = useState(null);
    const [status, setStatus] = useState({ running: false });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchResults();
        const interval = setInterval(checkStatus, 2000);
        return () => clearInterval(interval);
    }, []);

    const fetchResults = async () => {
        try {
            const response = await axios.get('http://localhost:8000/api/benchmarks/results');
            setResults(response.data);
            setLoading(false);
        } catch (err) {
            setLoading(false);
        }
    };

    const checkStatus = async () => {
        try {
            const response = await axios.get('http://localhost:8000/api/benchmarks/status');
            setStatus(response.data);

            // Refresh results when benchmark completes
            if (!response.data.running && status.running) {
                fetchResults();
            }
        } catch (err) {
            // Ignore
        }
    };

    const runBenchmark = async () => {
        setError(null);
        try {
            await axios.post('http://localhost:8000/api/benchmarks/run');
            setStatus({ running: true, progress: 0 });
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to start benchmark');
        }
    };

    const getBestModel = (metric) => {
        if (!results?.results?.length) return null;
        return results.results.reduce((best, curr) => {
            if (metric === 'tokens_per_second' || metric === 'fact_retention_score') {
                return curr[metric] > best[metric] ? curr : best;
            }
            return curr[metric] < best[metric] ? curr : best;
        });
    };

    return (
        <div className="space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-primary" />
                    <h3 className="font-semibold">Model Benchmarks</h3>
                </div>
                <button
                    onClick={runBenchmark}
                    disabled={status.running}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                    {status.running ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Running...
                        </>
                    ) : (
                        <>
                            <Play className="w-4 h-4" />
                            Run Benchmark
                        </>
                    )}
                </button>
            </div>

            {error && (
                <div className="p-2 rounded-lg bg-destructive/10 text-destructive text-sm">
                    {error}
                </div>
            )}

            {status.running && (
                <div className="p-3 rounded-lg border border-border bg-card">
                    <div className="flex items-center gap-2 text-sm mb-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Benchmarking models... This may take several minutes.</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2">
                        <div
                            className="bg-primary h-2 rounded-full transition-all duration-300"
                            style={{ width: `${status.progress || 5}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Results */}
            {results?.results?.length > 0 ? (
                <>
                    {/* Winners */}
                    <div className="grid grid-cols-3 gap-2">
                        <div className="p-2 rounded-lg border border-border bg-card text-center">
                            <Zap className="w-4 h-4 mx-auto mb-1 text-yellow-500" />
                            <p className="text-xs text-muted-foreground">Fastest</p>
                            <p className="text-sm font-medium truncate">
                                {getBestModel('tokens_per_second')?.model_name?.split(' ')[0]}
                            </p>
                        </div>
                        <div className="p-2 rounded-lg border border-border bg-card text-center">
                            <Brain className="w-4 h-4 mx-auto mb-1 text-purple-500" />
                            <p className="text-xs text-muted-foreground">Most Accurate</p>
                            <p className="text-sm font-medium truncate">
                                {getBestModel('fact_retention_score')?.model_name?.split(' ')[0]}
                            </p>
                        </div>
                        <div className="p-2 rounded-lg border border-border bg-card text-center">
                            <HardDrive className="w-4 h-4 mx-auto mb-1 text-green-500" />
                            <p className="text-xs text-muted-foreground">Most Efficient</p>
                            <p className="text-sm font-medium truncate">
                                {getBestModel('peak_memory_mb')?.model_name?.split(' ')[0]}
                            </p>
                        </div>
                    </div>

                    {/* Table */}
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-border">
                                    <th className="text-left py-2 px-1">Model</th>
                                    <th className="text-right py-2 px-1">TPS</th>
                                    <th className="text-right py-2 px-1">Accuracy</th>
                                    <th className="text-right py-2 px-1">Memory</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.results.map((r, idx) => (
                                    <tr key={idx} className="border-b border-border/50">
                                        <td className="py-2 px-1 font-medium truncate max-w-[120px]">
                                            {r.model_name}
                                        </td>
                                        <td className="text-right py-2 px-1 text-muted-foreground">
                                            {r.tokens_per_second?.toFixed(1)}
                                        </td>
                                        <td className="text-right py-2 px-1 text-muted-foreground">
                                            {r.fact_retention_score?.toFixed(0)}%
                                        </td>
                                        <td className="text-right py-2 px-1 text-muted-foreground">
                                            {r.peak_memory_mb?.toFixed(0)}MB
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <p className="text-xs text-muted-foreground text-center">
                        Last run: {results.timestamp || 'N/A'}
                    </p>
                </>
            ) : !loading && !status.running ? (
                <div className="text-center py-8 text-muted-foreground">
                    <BarChart3 className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No benchmark results yet</p>
                    <p className="text-xs">Download models and run a benchmark to compare performance</p>
                </div>
            ) : null}
        </div>
    );
};

export default BenchmarkResults;
