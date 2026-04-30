import React, { useState, useEffect } from 'react';
import axios from 'axios';

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
        <div className="space-y-10 animate-in fade-in duration-700">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h3 className="text-2xl font-bold font-headline flex items-center gap-3">
                        <span className="material-symbols-outlined text-primary">analytics</span>
                        Neural Performance
                    </h3>
                    <p className="text-xs opacity-60 font-medium mt-1 uppercase tracking-widest font-black">Benchmark local model efficiency</p>
                </div>
                <button
                    onClick={runBenchmark}
                    disabled={status.running}
                    className="flex items-center gap-3 px-8 py-3 rounded-full bg-primary text-white font-bold text-sm hover:shadow-xl hover:shadow-primary/20 transition-all active:scale-95 disabled:opacity-50"
                >
                    {status.running ? (
                        <>
                            <span className="material-symbols-outlined text-sm animate-spin">progress_activity</span>
                            Running Sequence...
                        </>
                    ) : (
                        <>
                            <span className="material-symbols-outlined text-sm">play_arrow</span>
                            Start Benchmark
                        </>
                    )}
                </button>
            </div>

            {error && (
                <div className="p-5 rounded-3xl bg-red-500/10 text-red-500 text-sm font-bold border border-red-500/20 flex items-center gap-3">
                    <span className="material-symbols-outlined">error</span>
                    {error}
                </div>
            )}

            {status.running && (
                <div className="p-8 rounded-[2.5rem] bg-primary/5 border border-primary/20 space-y-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <span className="material-symbols-outlined animate-spin text-primary">query_stats</span>
                            <span className="font-bold">Analyzing Neural Weights...</span>
                        </div>
                        <span className="font-black text-primary">{status.progress || 5}%</span>
                    </div>
                    <div className="w-full bg-primary/10 rounded-full h-3 overflow-hidden">
                        <div
                            className="bg-primary h-full rounded-full transition-all duration-500"
                            style={{ width: `${status.progress || 5}%` }}
                        />
                    </div>
                    <p className="text-[10px] font-black uppercase tracking-widest opacity-40">This sequence can take several minutes</p>
                </div>
            )}

            {/* Results */}
            {results?.results?.length > 0 ? (
                <>
                    {/* Top Performers Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="p-8 rounded-[2.5rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm text-center space-y-3 group hover:border-amber-500/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-amber-500/10 text-amber-500 mx-auto flex items-center justify-center">
                                <span className="material-symbols-outlined text-2xl">bolt</span>
                            </div>
                            <p className="text-[10px] font-black uppercase opacity-40 tracking-widest">Fastest Throughput</p>
                            <p className="text-xl font-bold font-headline truncate">
                                {getBestModel('tokens_per_second')?.model_name?.split(' ')[0]}
                            </p>
                        </div>
                        <div className="p-8 rounded-[2.5rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm text-center space-y-3 group hover:border-primary/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-primary/10 text-primary mx-auto flex items-center justify-center">
                                <span className="material-symbols-outlined text-2xl">psychology</span>
                            </div>
                            <p className="text-[10px] font-black uppercase opacity-40 tracking-widest">Maximum Accuracy</p>
                            <p className="text-xl font-bold font-headline truncate">
                                {getBestModel('fact_retention_score')?.model_name?.split(' ')[0]}
                            </p>
                        </div>
                        <div className="p-8 rounded-[2.5rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm text-center space-y-3 group hover:border-emerald-500/30 transition-all">
                            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 text-emerald-500 mx-auto flex items-center justify-center">
                                <span className="material-symbols-outlined text-2xl">memory</span>
                            </div>
                            <p className="text-[10px] font-black uppercase opacity-40 tracking-widest">Hardware Efficient</p>
                            <p className="text-xl font-bold font-headline truncate">
                                {getBestModel('peak_memory_mb')?.model_name?.split(' ')[0]}
                            </p>
                        </div>
                    </div>

                    {/* Full Comparison Table */}
                    <div className="bg-white dark:bg-slate-900 rounded-[3rem] border border-[#f3f3fd] dark:border-slate-800 overflow-hidden shadow-sm">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="bg-[#f3f3fd] dark:bg-slate-950/40">
                                    <th className="py-6 px-8 text-xs font-black uppercase tracking-widest opacity-40">Neural Model</th>
                                    <th className="py-6 px-8 text-xs font-black uppercase tracking-widest opacity-40 text-right">Throughput (TPS)</th>
                                    <th className="py-6 px-8 text-xs font-black uppercase tracking-widest opacity-40 text-right">Precision</th>
                                    <th className="py-6 px-8 text-xs font-black uppercase tracking-widest opacity-40 text-right">VRAM Peak</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-[#f3f3fd] dark:divide-slate-800">
                                {results.results.map((r, idx) => (
                                    <tr key={idx} className="hover:bg-primary/5 transition-colors">
                                        <td className="py-6 px-8">
                                            <p className="font-bold text-sm">{r.model_name}</p>
                                        </td>
                                        <td className="py-6 px-8 text-right font-mono text-sm font-bold text-amber-500">
                                            {r.tokens_per_second?.toFixed(1)}
                                        </td>
                                        <td className="py-6 px-8 text-right font-mono text-sm font-bold text-primary">
                                            {r.fact_retention_score?.toFixed(0)}%
                                        </td>
                                        <td className="py-6 px-8 text-right font-mono text-sm font-bold opacity-60">
                                            {r.peak_memory_mb?.toFixed(0)}MB
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div className="flex items-center justify-center gap-3 opacity-30 text-[10px] font-black uppercase tracking-widest">
                        <span className="material-symbols-outlined text-sm">schedule</span>
                        Last Telemetry: {results.timestamp || 'N/A'}
                    </div>
                </>
            ) : !loading && !status.running ? (
                <div className="p-24 rounded-[3rem] bg-[#f3f3fd] dark:bg-slate-950/40 border-2 border-dashed border-[#d1d1f0] dark:border-slate-800 text-center space-y-6">
                    <div className="w-24 h-24 rounded-[2rem] bg-white dark:bg-slate-900 shadow-xl mx-auto flex items-center justify-center text-[#d1d1f0]">
                        <span className="material-symbols-outlined text-5xl">bar_chart_4_bars</span>
                    </div>
                    <div>
                        <p className="text-xl font-bold font-headline">Telemetry Database Empty</p>
                        <p className="text-sm opacity-60 font-medium">Download local models and execute a performance scan to generate data.</p>
                    </div>
                </div>
            ) : null}
        </div>
    );
};

export default BenchmarkResults;
