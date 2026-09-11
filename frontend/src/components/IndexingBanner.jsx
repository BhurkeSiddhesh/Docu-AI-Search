import React, { useEffect, useRef, useState } from 'react';
import { Loader2, CheckCircle2, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { motion } from 'framer-motion';
import api from '../lib/api';

// Human-readable labels for each pipeline stage
const STAGE_LABELS = {
    starting:    'Starting…',
    extracting:  'Extracting',
    chunking:    'Chunking',
    embedding:   'Embedding',
    indexing:    'Indexing',
    finalizing:  'Finalizing',
};

const STAGE_ORDER = ['starting', 'extracting', 'chunking', 'embedding', 'indexing', 'finalizing'];

const PIPELINE_STEPS = [
    { label: 'Read', stages: ['starting', 'extracting'] },
    { label: 'Chunk', stages: ['chunking'] },
    { label: 'Embed', stages: ['embedding'] },
    { label: 'Index', stages: ['indexing', 'finalizing'] },
];

function pipelineStepState(stepIndex, stage, progress, complete = false) {
    if (complete) return 'done';
    const activeIndex = Math.max(
        0,
        PIPELINE_STEPS.findIndex((step) => step.stages.includes(stage))
    );
    if (stage) {
        if (stepIndex < activeIndex) return 'done';
        if (stepIndex === activeIndex) return 'active';
        return 'pending';
    }
    const progressIndex = Math.min(PIPELINE_STEPS.length - 1, Math.floor((progress || 0) / 25));
    if (stepIndex < progressIndex) return 'done';
    if (stepIndex === progressIndex) return 'active';
    return 'pending';
}

function IndexingPipeline({ stage, progress, complete = false }) {
    return (
        <div className="mt-3" data-testid="indexing-pipeline">
            <div className="grid grid-cols-4 gap-2">
                {PIPELINE_STEPS.map((step, index) => {
                    const state = pipelineStepState(index, stage, progress, complete);
                    return (
                        <div key={step.label} className="relative">
                            {index > 0 && (
                                <div className="absolute right-1/2 top-3 h-px w-full bg-[rgba(0,112,243,0.16)] dark:bg-[rgba(50,145,255,0.18)]" />
                            )}
                            <div className="relative z-10 flex flex-col items-center gap-1">
                                <motion.span
                                    className={`w-6 h-6 rounded-full border flex items-center justify-center text-[9px] font-mono ${
                                        state === 'done'
                                            ? 'bg-link text-white border-link dark:bg-[#3291ff] dark:border-[#3291ff]'
                                            : state === 'active'
                                                ? 'bg-canvas text-link border-link dark:bg-[#0a0a0a] dark:text-[#3291ff] dark:border-[#3291ff]'
                                                : 'bg-canvas text-mute border-hairline dark:bg-[#0a0a0a] dark:border-[rgba(255,255,255,0.12)]'
                                    }`}
                                    animate={state === 'active' ? { scale: [1, 1.12, 1] } : { scale: 1 }}
                                    transition={{ duration: 1.1, repeat: state === 'active' ? Infinity : 0, ease: 'easeInOut' }}
                                >
                                    {state === 'done' ? 'OK' : index + 1}
                                </motion.span>
                                <span className={`font-mono text-[10px] uppercase tracking-[0.05em] ${
                                    state === 'pending' ? 'text-mute' : 'text-link dark:text-[#3291ff]'
                                }`}>
                                    {step.label}
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

function formatEta(seconds) {
    if (seconds == null || seconds <= 0) return null;
    if (seconds < 60) return `~${Math.round(seconds)}s remaining`;
    const m = Math.round(seconds / 60);
    return `~${m} min remaining`;
}

function formatFileCount(processed, total) {
    if (!total || total <= 0) return null;
    return `${processed ?? 0} / ${total} files`;
}

export default function IndexingBanner() {
    const [status, setStatus] = useState({
        running: false,
        progress: 0,
        current_file: '',
        stage: null,
        total_files: 0,
        processed_files: 0,
        eta_seconds: null,
        error: null,
    });
    const [recentlyDone, setRecentlyDone] = useState(false);
    const [expanded, setExpanded] = useState(false);
    const wasRunning = useRef(false);

    useEffect(() => {
        let ws;
        let closed = false;
        let reconnectTimer;

        const connect = () => {
            if (closed) return;
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/progress`);

            ws.onmessage = (evt) => {
                try {
                    const data = JSON.parse(evt.data);
                    if (data.type === 'indexing_progress') {
                        const next = {
                            running: true,
                            progress: data.percent ?? 0,
                            current_file: data.current_file ?? '',
                            stage: data.stage ?? null,
                            total_files: data.total_files ?? 0,
                            processed_files: data.processed_files ?? 0,
                            eta_seconds: data.eta_seconds ?? null,
                            error: null,
                        };
                        wasRunning.current = true;
                        setStatus(next);
                    } else if (data.type === 'indexing_complete') {
                        if (wasRunning.current) {
                            setRecentlyDone(true);
                            setTimeout(() => setRecentlyDone(false), 4000);
                        }
                        wasRunning.current = false;
                        setStatus({ running: false, progress: 100, current_file: '', stage: null, total_files: 0, processed_files: 0, eta_seconds: null, error: null });
                    } else if (data.type === 'error') {
                        wasRunning.current = false;
                        setStatus((s) => ({ ...s, running: false, error: data.message ?? 'Unknown error' }));
                    }
                } catch {
                    // ignore parse errors
                }
            };

            ws.onclose = () => {
                if (!closed) {
                    pollOnce();
                    reconnectTimer = setTimeout(connect, 3000);
                }
            };

            ws.onerror = () => { ws.close(); };
        };

        const pollOnce = async () => {
            if (closed) return;
            try {
                const res = await api.getIndexStatus();
                const data = res.data || {};
                if (wasRunning.current && !data.running && !data.error) {
                    setRecentlyDone(true);
                    setTimeout(() => setRecentlyDone(false), 4000);
                }
                wasRunning.current = !!data.running;
                setStatus(data);
            } catch {
                // ignore
            }
        };

        pollOnce().then(() => { if (!closed) connect(); });

        return () => {
            closed = true;
            clearTimeout(reconnectTimer);
            if (ws) ws.close();
        };
    }, []);

    // ── Error state ────────────────────────────────────────────────────────
    if (status.error) {
        return (
            <div className="flex items-center gap-3 bg-error-soft dark:bg-[rgba(238,0,0,0.1)] border border-[rgba(238,0,0,0.2)] rounded-v-md px-4 py-3 text-sm">
                <AlertCircle className="w-4 h-4 text-error-deep dark:text-error flex-shrink-0" />
                <div className="flex-1 min-w-0">
                    <div className="font-semibold text-error-deep dark:text-[#ffcccc] tracking-[-0.28px]">Indexing failed</div>
                    <div className="text-[13px] text-error dark:text-[#ff9999] truncate mt-0.5">{status.error}</div>
                </div>
            </div>
        );
    }

    if (!status.running && !recentlyDone) return null;

    // ── Complete state ─────────────────────────────────────────────────────
    if (recentlyDone) {
        return (
            <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="bg-[#e6f4ea] dark:bg-[rgba(52,168,83,0.1)] border border-[#ceead6] dark:border-[rgba(52,168,83,0.2)] rounded-v-md px-4 py-3 text-sm overflow-hidden"
            >
                <div className="flex items-center gap-3">
                    <CheckCircle2 className="w-4 h-4 text-[#188038] dark:text-[#81c995] flex-shrink-0" />
                    <span className="font-semibold text-[#137333] dark:text-[#81c995] tracking-[-0.28px]">Indexing complete</span>
                </div>
                <IndexingPipeline progress={100} complete />
            </motion.div>
        );
    }

    // ── Running state ──────────────────────────────────────────────────────
    const stageLabel = STAGE_LABELS[status.stage] ?? 'Processing…';
    const fileCount  = formatFileCount(status.processed_files, status.total_files);
    const eta        = formatEta(status.eta_seconds);
    const pct        = status.progress ?? 0;
    const isAtZero   = pct === 0;

    // Stage pill colours
    const stagePillClass = status.stage === 'embedding'
        ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300'
        : status.stage === 'extracting'
        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
        : status.stage === 'finalizing'
        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
        : 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300';

    return (
        <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-link-bg-soft dark:bg-[rgba(0,112,243,0.08)] border border-[rgba(0,112,243,0.15)] dark:border-[rgba(0,112,243,0.2)] rounded-v-md px-4 py-3 shadow-v-2 dark:shadow-v-dark-2"
        >

            {/* ── Header row ─────────────────────────────────────────────── */}
            <div className="flex items-center gap-2.5">
                <Loader2 className="w-4 h-4 text-link dark:text-[#3291ff] animate-spin flex-shrink-0" />

                {/* Stage chip */}
                <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold tracking-wide ${stagePillClass}`}>
                    {stageLabel}
                </span>

                {/* Percent */}
                <span className="font-semibold text-sm text-link-deep dark:text-[#3291ff] tracking-[-0.28px] tabular-nums">
                    {pct}%
                </span>

                {/* File counter */}
                {fileCount && (
                    <span className="text-[12px] text-link dark:text-[rgba(50,145,255,0.7)] tabular-nums hidden sm:inline">
                        · {fileCount}
                    </span>
                )}

                {/* ETA — push to right */}
                {eta && (
                    <span className="ml-auto text-[11px] text-link dark:text-[rgba(50,145,255,0.6)] tabular-nums hidden md:inline whitespace-nowrap">
                        {eta}
                    </span>
                )}

                {/* Expand toggle */}
                <button
                    onClick={() => setExpanded(e => !e)}
                    className="ml-auto md:ml-0 p-0.5 rounded text-link/60 dark:text-[rgba(50,145,255,0.5)] hover:text-link dark:hover:text-[#3291ff] transition-colors"
                    aria-label={expanded ? 'Collapse detail' : 'Expand detail'}
                >
                    {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>
            </div>

            {/* ── Progress bar ────────────────────────────────────────────── */}
            <div className="mt-2.5 h-1.5 w-full bg-[rgba(0,112,243,0.15)] dark:bg-[rgba(0,112,243,0.2)] rounded-full overflow-hidden relative">
                {/* Shimmer animation when at 0% so user knows it's running */}
                {isAtZero ? (
                    <div className="h-full w-full relative overflow-hidden rounded-full">
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[rgba(0,112,243,0.4)] to-transparent animate-[shimmer_1.5s_ease-in-out_infinite] -translate-x-full" />
                    </div>
                ) : (
                    <div
                        className="h-full bg-gradient-to-r from-gradient-develop-start to-gradient-develop-end transition-all duration-500 ease-out"
                        style={{ width: `${pct}%` }}
                    />
                )}
            </div>

            {/* ── Expanded detail: current file path ──────────────────────── */}
            {expanded && status.current_file && (
                <div className="mt-2 text-[11px] font-mono text-link dark:text-[rgba(50,145,255,0.7)] break-all leading-relaxed">
                    {status.current_file}
                </div>
            )}

            {/* ── Collapsed: compact single-line filename ─────────────────── */}
            {!expanded && status.current_file && (
                <div className="mt-1.5 text-[11px] font-mono text-link dark:text-[rgba(50,145,255,0.7)] truncate">
                    {status.current_file}
                </div>
            )}
            <IndexingPipeline stage={status.stage} progress={pct} />
        </motion.div>
    );
}
