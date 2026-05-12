import React, { useEffect, useRef, useState } from 'react';
import { Loader2, CheckCircle2, AlertCircle } from 'lucide-react';
import api from '../lib/api';

export default function IndexingBanner() {
    const [status, setStatus] = useState({ running: false, progress: 0, current_file: '', error: null });
    const [recentlyDone, setRecentlyDone] = useState(false);
    const wasRunning = useRef(false);

    useEffect(() => {
        let cancelled = false;
        let timer;

        const tick = async () => {
            try {
                const res = await api.getIndexStatus();
                if (cancelled) return;
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
            if (!cancelled) timer = setTimeout(tick, 1500);
        };

        tick();
        return () => {
            cancelled = true;
            clearTimeout(timer);
        };
    }, []);

    if (status.error) {
        return (
            <div className="flex items-center gap-3 bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 rounded-lg px-4 py-3 text-sm">
                <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                    <div className="font-medium text-red-900 dark:text-red-200">Indexing failed</div>
                    <div className="text-xs text-red-700 dark:text-red-300 truncate">{status.error}</div>
                </div>
            </div>
        );
    }

    if (!status.running && !recentlyDone) return null;

    if (recentlyDone) {
        return (
            <div className="flex items-center gap-3 bg-green-50 dark:bg-green-950/40 border border-green-200 dark:border-green-900 rounded-lg px-4 py-3 text-sm animate-fade-in">
                <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                <span className="font-medium text-green-900 dark:text-green-200">Indexing complete</span>
            </div>
        );
    }

    return (
        <div className="bg-primary/5 dark:bg-primary/10 border border-primary/20 rounded-lg px-4 py-3">
            <div className="flex items-center gap-3 mb-2">
                <Loader2 className="w-4 h-4 text-primary animate-spin flex-shrink-0" />
                <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm text-slate-900 dark:text-slate-100">
                        Indexing documents… {status.progress}%
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 truncate">
                        {status.current_file || 'Scanning…'}
                    </div>
                </div>
            </div>
            <div className="h-1.5 w-full bg-slate-200 dark:bg-slate-800 rounded-full overflow-hidden">
                <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${status.progress}%` }}
                />
            </div>
        </div>
    );
}
