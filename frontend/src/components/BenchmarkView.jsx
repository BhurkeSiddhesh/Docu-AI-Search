import React, { useEffect, useState } from 'react';
import { Play, Loader2, RefreshCw, Zap, HardDrive, Clock, Trophy, Target } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';

function medal(rank) {
    if (rank === 0) return '🥇';
    if (rank === 1) return '🥈';
    if (rank === 2) return '🥉';
    return `#${rank + 1}`;
}

const fmt = (v, suffix = '') =>
    v == null || Number.isNaN(v) ? '—' : `${(typeof v === 'number' ? v.toFixed(1) : v)}${suffix}`;

export default function BenchmarkView() {
    const [results, setResults] = useState(null);
    const [status, setStatus] = useState({ running: false, progress: 0 });
    const [loading, setLoading] = useState(true);
    const toast = useToast();

    useEffect(() => {
        load();
        const t = setInterval(pollStatus, 2000);
        return () => clearInterval(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const load = async () => {
        setLoading(true);
        try {
            const r = await api.benchmarkResults();
            setResults(r.data);
        } catch {
            // ignore
        } finally {
            setLoading(false);
        }
    };

    const pollStatus = async () => {
        try {
            const r = await api.benchmarkStatus();
            const wasRunning = status.running;
            setStatus(r.data);
            if (wasRunning && !r.data.running) load();
        } catch {
            // ignore
        }
    };

    const run = async () => {
        try {
            await api.runBenchmarks();
            toast.info('Benchmark started');
            setStatus({ running: true, progress: 0 });
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not start benchmark');
        }
    };

    const ranked = (results?.results || [])
        .slice()
        .sort((a, b) => (b.weighted_score ?? 0) - (a.weighted_score ?? 0));

    return (
        <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 lg:py-10">
            <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50 mb-1">Benchmarks</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                        Compare speed, memory, and accuracy across your local models.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={load} className="btn-secondary" disabled={loading}>
                        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                        Refresh
                    </button>
                    <button onClick={run} disabled={status.running} className="btn-primary">
                        {status.running
                            ? <><Loader2 className="w-4 h-4 animate-spin" /> Running ({status.progress}%)</>
                            : <><Play className="w-4 h-4" /> Run benchmark</>}
                    </button>
                </div>
            </div>

            {status.running && (
                <div className="card p-4 mb-5 bg-primary/5 border-primary/20">
                    <div className="flex items-center gap-3 mb-2">
                        <Loader2 className="w-4 h-4 text-primary animate-spin" />
                        <div className="text-sm font-medium">
                            Benchmarking {status.current_model || '…'} — {status.progress}%
                        </div>
                    </div>
                    <div className="h-1.5 bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-primary transition-all" style={{ width: `${status.progress}%` }} />
                    </div>
                </div>
            )}

            {loading ? (
                <div className="card p-10 text-center text-slate-500 dark:text-slate-400 text-sm">Loading…</div>
            ) : ranked.length === 0 ? (
                <div className="card p-10 text-center">
                    <Trophy className="w-10 h-10 mx-auto mb-3 text-slate-300 dark:text-slate-700" />
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">No benchmark results yet</p>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">Run a benchmark to evaluate your downloaded local models.</p>
                </div>
            ) : (
                <>
                    {results?.timestamp && (
                        <div className="text-xs text-slate-500 dark:text-slate-400 mb-3">
                            Last run: {results.timestamp}
                        </div>
                    )}
                    <div className="card overflow-hidden">
                        <div className="grid grid-cols-12 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 px-4 py-3 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/50">
                            <div className="col-span-1">Rank</div>
                            <div className="col-span-4">Model</div>
                            <div className="col-span-2 text-right">Score</div>
                            <div className="col-span-2 text-right">Speed</div>
                            <div className="col-span-2 text-right">Accuracy</div>
                            <div className="col-span-1 text-right">Load</div>
                        </div>
                        <div className="divide-y divide-slate-200 dark:divide-slate-800">
                            {ranked.map((r, i) => (
                                <div key={r.model_path || r.model_name || i} className="grid grid-cols-12 items-center px-4 py-3 text-sm hover:bg-slate-50 dark:hover:bg-slate-800/40 transition">
                                    <div className="col-span-1 text-lg">{medal(i)}</div>
                                    <div className="col-span-4 font-medium truncate text-slate-900 dark:text-slate-50" title={r.model_name}>
                                        {r.model_name}
                                    </div>
                                    <div className="col-span-2 text-right font-mono text-slate-700 dark:text-slate-300">
                                        {fmt(r.weighted_score)}
                                    </div>
                                    <div className="col-span-2 text-right font-mono text-slate-700 dark:text-slate-300 flex items-center justify-end gap-1">
                                        <Zap className="w-3 h-3 text-amber-500" />
                                        {fmt(r.tokens_per_second)}
                                        <span className="text-[10px] text-slate-400 ml-0.5">TPS</span>
                                    </div>
                                    <div className="col-span-2 text-right font-mono text-slate-700 dark:text-slate-300 flex items-center justify-end gap-1">
                                        <Target className="w-3 h-3 text-emerald-500" />
                                        {fmt(r.fact_retention_score, '%')}
                                    </div>
                                    <div className="col-span-1 text-right font-mono text-slate-700 dark:text-slate-300 flex items-center justify-end gap-1">
                                        <Clock className="w-3 h-3 text-slate-400" />
                                        {fmt(r.load_time_s, 's')}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Memory secondary row */}
                    <div className="mt-3 text-xs text-slate-500 dark:text-slate-400 flex items-center gap-1">
                        <HardDrive className="w-3 h-3" />
                        Memory usage:{' '}
                        {ranked.map((r, i) => (
                            <span key={i} className="font-mono ml-2">
                                {r.model_name?.split(' ')[0]}: {fmt(r.peak_memory_mb, 'MB')}
                                {i < ranked.length - 1 ? ' ·' : ''}
                            </span>
                        ))}
                    </div>
                </>
            )}
        </div>
    );
}
