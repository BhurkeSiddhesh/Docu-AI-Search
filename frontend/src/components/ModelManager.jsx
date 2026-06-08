import React, { useEffect, useState } from 'react';
import { Download, Cpu, Trash2, Check, Loader2, Star } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';
import { formatBytes } from '../lib/format';

export default function ModelManager({ activeModelPath, onSelectModel }) {
    const [available, setAvailable] = useState([]);
    const [local, setLocal] = useState([]);
    const [downloadStatus, setDownloadStatus] = useState({ downloading: false });
    const [filter, setFilter] = useState('all');
    const [query, setQuery] = useState('');
    const toast = useToast();

    useEffect(() => {
        load();
        const t = setInterval(pollStatus, 2000);
        return () => clearInterval(t);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const load = async () => {
        try {
            const [a, l] = await Promise.all([api.listAvailableModels(), api.listLocalModels()]);
            setAvailable(a.data || []);
            setLocal(l.data || []);
        } catch (e) {
            toast.error('Could not load model list — is the backend running?');
            console.error('ModelManager load error:', e);
        }
    };

    const pollStatus = async () => {
        try {
            const r = await api.modelDownloadStatus();
            const wasDownloading = downloadStatus.downloading;
            setDownloadStatus(r.data);
            if (wasDownloading && !r.data.downloading) load();
        } catch (e) {
            console.warn('ModelManager poll error:', e);
        }
    };

    const download = async (id) => {
        try {
            await api.downloadModel(id);
            setDownloadStatus({ downloading: true, model_id: id, progress: 0 });
            toast.info('Download started');
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Download failed');
        }
    };

    const remove = async (path) => {
        if (!confirm('Delete this model file?')) return;
        try {
            await api.deleteModel(path);
            toast.success('Model deleted');
            load();
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not delete model');
        }
    };

    const isDownloaded = (id) => local.some((m) => m.id === id || (m.filename || '').includes(id));

    const norm = (s) => (s || '').replace(/\\/g, '/').toLowerCase();
    const filtered = available.filter((m) => {
        if (filter !== 'all' && m.category !== filter) return false;
        if (query && !`${m.name} ${m.description}`.toLowerCase().includes(query.toLowerCase())) return false;
        return true;
    });

    return (
        <div className="space-y-6">
            {/* Local */}
            <section>
                <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-slate-900 dark:text-slate-50">Downloaded models</h3>
                    <span className="chip text-xs">{local.length} installed</span>
                </div>

                {local.length === 0 ? (
                    <div className="card p-6 text-center text-sm text-slate-500 dark:text-slate-400">
                        No local models. Download one from the registry below.
                    </div>
                ) : (
                    <div className="grid sm:grid-cols-2 gap-3">
                        {local.map((m, i) => {
                            const isActive = activeModelPath && norm(m.path) === norm(activeModelPath);
                            return (
                                <div
                                    key={i}
                                    className={`card p-4 transition ${isActive ? 'border-primary ring-2 ring-primary/20' : ''}`}
                                >
                                    <div className="flex items-start gap-3 mb-3">
                                        <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${isActive ? 'bg-primary text-white' : 'bg-primary/10 text-primary'}`}>
                                            <Cpu className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="font-semibold text-sm text-slate-900 dark:text-slate-50 truncate" title={m.name}>{m.name}</div>
                                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                                {m.size ? formatBytes(m.size) : '—'}
                                                {m.ram_required && ` · ${m.ram_required} GB RAM`}
                                            </div>
                                        </div>
                                        {isActive && (
                                            <span className="chip text-[10px] bg-primary/10 text-primary">Active</span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => onSelectModel?.(m)}
                                            disabled={isActive}
                                            className={`flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium transition ${
                                                isActive
                                                    ? 'bg-slate-100 dark:bg-slate-800 text-slate-500 cursor-not-allowed'
                                                    : 'bg-primary text-white hover:bg-primary/90'
                                            }`}
                                        >
                                            {isActive ? <><Check className="w-3.5 h-3.5" /> Selected</> : 'Select'}
                                        </button>
                                        <button
                                            onClick={() => remove(m.path)}
                                            title="Delete model"
                                            className="p-2 rounded-md text-slate-500 hover:bg-red-50 dark:hover:bg-red-950/40 hover:text-red-500 transition"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </section>

            {/* Registry */}
            <section>
                <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-slate-900 dark:text-slate-50">Available models</h3>
                </div>

                <div className="flex flex-wrap gap-2 mb-3">
                    {['all', 'small', 'medium', 'large'].map((c) => (
                        <button
                            key={c}
                            onClick={() => setFilter(c)}
                            className={`px-3 py-1.5 rounded-md text-xs font-medium capitalize transition ${
                                filter === c
                                    ? 'bg-primary text-white'
                                    : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                            }`}
                        >
                            {c === 'all' ? 'All' : c}
                        </button>
                    ))}
                    <input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search models…"
                        className="input flex-1 min-w-[180px] py-1.5 text-xs"
                    />
                </div>

                <div className="space-y-2">
                    {filtered.map((m) => {
                        const ready = isDownloaded(m.id);
                        const isDownloading = downloadStatus.downloading && downloadStatus.model_id === m.id;
                        return (
                            <div key={m.id} className="card p-4 flex items-center gap-4">
                                <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${m.recommended ? 'bg-amber-100 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-500'}`}>
                                    {m.recommended ? <Star className="w-4 h-4" /> : <Cpu className="w-4 h-4" />}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-0.5">
                                        <div className="font-semibold text-sm text-slate-900 dark:text-slate-50 truncate">{m.name}</div>
                                        {m.recommended && <span className="chip text-[10px] bg-amber-100 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400">Recommended</span>}
                                    </div>
                                    <div className="text-xs text-slate-500 dark:text-slate-400 truncate mb-1">{m.description}</div>
                                    <div className="text-[11px] text-slate-400 dark:text-slate-500 font-mono">
                                        {m.size} · {m.ram_required}GB RAM · {m.quantization}
                                    </div>
                                </div>
                                <div className="flex-shrink-0">
                                    {ready ? (
                                        <span className="inline-flex items-center gap-1 text-xs font-medium text-green-600 dark:text-green-400">
                                            <Check className="w-3.5 h-3.5" />
                                            Installed
                                        </span>
                                    ) : isDownloading ? (
                                        <div className="flex flex-col items-end gap-1 w-28">
                                            <div className="text-xs font-mono text-slate-600 dark:text-slate-400">
                                                {downloadStatus.progress || 0}%
                                            </div>
                                            <div className="h-1 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                                <div className="h-full bg-primary transition-all" style={{ width: `${downloadStatus.progress || 0}%` }} />
                                            </div>
                                        </div>
                                    ) : (
                                        <button
                                            onClick={() => download(m.id)}
                                            disabled={downloadStatus.downloading}
                                            className="btn-secondary text-xs py-1.5 px-3"
                                        >
                                            <Download className="w-3.5 h-3.5" />
                                            Download
                                        </button>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                    {filtered.length === 0 && (
                        <div className="text-center py-8 text-sm text-slate-500 dark:text-slate-400">No models match your filter.</div>
                    )}
                </div>
            </section>
        </div>
    );
}
