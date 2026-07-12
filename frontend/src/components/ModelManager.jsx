import React, { useEffect, useRef, useState } from 'react';
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
    const prevDownloadingRef = useRef(false);
    const pollStatusRef = useRef(null);
    const toast = useToast();

    useEffect(() => {
        load();
        const t = setInterval(() => pollStatusRef.current?.(), 2000);
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
            setDownloadStatus((prev) => {
                if (prev.downloading && !r.data.downloading) {
                    setTimeout(load, 0);
                }
                return r.data;
            });
        } catch (e) {
            console.warn('ModelManager poll error:', e);
        }
    };
    pollStatusRef.current = pollStatus;

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
                    <h3 className="font-semibold text-ink dark:text-[#ededed] tracking-[-0.28px]">Downloaded models</h3>
                    <span className="chip text-xs">{local.length} installed</span>
                </div>

                {local.length === 0 ? (
                    <div className="card p-6 text-center text-sm text-mute">
                        No local models. Download one from the registry below.
                    </div>
                ) : (
                    <div className="grid sm:grid-cols-2 gap-3">
                        {local.map((m, i) => {
                            const isActive = activeModelPath && norm(m.path) === norm(activeModelPath);
                            return (
                                <div
                                    key={i}
                                    className={`card p-4 transition ${isActive ? 'border-ink dark:border-[#ededed] shadow-v-3 dark:shadow-v-dark-3' : ''}`}
                                >
                                    <div className="flex items-start gap-3 mb-3">
                                        <div className={`w-9 h-9 rounded-v-sm flex items-center justify-center flex-shrink-0 ${isActive ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink' : 'bg-canvas-soft dark:bg-[rgba(255,255,255,0.06)] text-ink dark:text-[#ededed]'}`}>
                                            <Cpu className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <div className="font-semibold text-sm text-ink dark:text-[#ededed] truncate tracking-[-0.28px]" title={m.name}>{m.name}</div>
                                            <div className="text-xs text-mute font-mono">
                                                {m.size ? formatBytes(m.size) : '—'}
                                                {m.ram_required && ` · ${m.ram_required} GB RAM`}
                                            </div>
                                        </div>
                                        {isActive && (
                                            <span className="badge text-[10px] bg-ink dark:bg-[#ededed] text-white dark:text-ink font-medium">Active</span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => onSelectModel?.(m)}
                                            disabled={isActive}
                                            className={`flex-1 btn-sm ${
                                                isActive
                                                    ? 'btn-secondary opacity-50 cursor-not-allowed'
                                                    : 'btn-primary'
                                            }`}
                                        >
                                            {isActive ? <><Check className="w-3.5 h-3.5" /> Selected</> : 'Select'}
                                        </button>
                                        <button
                                            onClick={() => remove(m.path)}
                                            title="Delete model"
                                            className="p-2 rounded-v-sm text-mute hover:bg-error-soft dark:hover:bg-[rgba(238,0,0,0.1)] hover:text-error transition"
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
                    <h3 className="font-semibold text-ink dark:text-[#ededed] tracking-[-0.28px]">Available models</h3>
                </div>

                <div className="flex flex-wrap gap-2 mb-3">
                    {['all', 'small', 'medium', 'large'].map((c) => (
                        <button
                            key={c}
                            onClick={() => setFilter(c)}
                            className={`px-3 py-1.5 rounded-pill-sm text-xs font-medium capitalize transition ${
                                filter === c
                                    ? 'bg-ink dark:bg-[#ededed] text-white dark:text-ink'
                                    : 'bg-canvas-soft dark:bg-[rgba(255,255,255,0.06)] text-body dark:text-[#888] border border-hairline dark:border-[rgba(255,255,255,0.1)] hover:border-hairline-strong dark:hover:border-[rgba(255,255,255,0.2)]'
                            }`}
                        >
                            {c === 'all' ? 'All' : c}
                        </button>
                    ))}
                    <input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search models..."
                        className="input flex-1 min-w-[180px] h-8 text-xs"
                    />
                </div>

                <div className="space-y-2">
                    {filtered.map((m) => {
                        const ready = isDownloaded(m.id);
                        const isDownloading = downloadStatus.downloading && downloadStatus.model_id === m.id;
                        return (
                            <div key={m.id} className="card p-4 flex items-center gap-4">
                                <div className={`w-9 h-9 rounded-v-sm flex items-center justify-center flex-shrink-0 ${m.recommended ? 'bg-warning-soft dark:bg-[rgba(245,166,35,0.15)] text-warning-deep dark:text-warning' : 'bg-canvas-soft dark:bg-[rgba(255,255,255,0.06)] text-mute'}`}>
                                    {m.recommended ? <Star className="w-4 h-4" /> : <Cpu className="w-4 h-4" />}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 mb-0.5">
                                        <div className="font-semibold text-sm text-ink dark:text-[#ededed] truncate tracking-[-0.28px]">{m.name}</div>
                                        {m.recommended && <span className="badge text-[10px] bg-warning-soft dark:bg-[rgba(245,166,35,0.15)] text-warning-deep dark:text-warning">Recommended</span>}
                                    </div>
                                    <div className="text-xs text-mute truncate mb-1">{m.description}</div>
                                    <div className="text-[11px] text-mute font-mono">
                                        {m.size} · {m.ram_required}GB RAM · {m.quantization}
                                    </div>
                                </div>
                                <div className="flex-shrink-0">
                                    {ready ? (
                                        <span className="inline-flex items-center gap-1 text-xs font-medium text-link">
                                            <Check className="w-3.5 h-3.5" />
                                            Installed
                                        </span>
                                    ) : isDownloading ? (
                                        <div className="flex flex-col items-end gap-1 w-28">
                                            <div className="text-xs font-mono text-mute">
                                                {downloadStatus.progress || 0}%
                                            </div>
                                            <div className="h-1 w-full bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.06)] rounded-full overflow-hidden">
                                                <div className="h-full bg-ink dark:bg-[#ededed] transition-all" style={{ width: `${downloadStatus.progress || 0}%` }} />
                                            </div>
                                        </div>
                                    ) : (
                                        <button
                                            onClick={() => download(m.id)}
                                            disabled={downloadStatus.downloading}
                                            className="btn-secondary btn-sm"
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
                        <div className="text-center py-8 text-sm text-mute">No models match your filter.</div>
                    )}
                </div>
            </section>
        </div>
    );
}
