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

    if (recentlyDone) {
        return (
            <div className="flex items-center gap-3 bg-[#e6f4ea] dark:bg-[rgba(52,168,83,0.1)] border border-[#ceead6] dark:border-[rgba(52,168,83,0.2)] rounded-v-md px-4 py-3 text-sm animate-fade-in">
                <CheckCircle2 className="w-4 h-4 text-[#188038] dark:text-[#81c995] flex-shrink-0" />
                <span className="font-semibold text-[#137333] dark:text-[#81c995] tracking-[-0.28px]">Indexing complete</span>
            </div>
        );
    }

    return (
        <div className="bg-link-bg-soft dark:bg-[rgba(0,112,243,0.08)] border border-[rgba(0,112,243,0.15)] dark:border-[rgba(0,112,243,0.2)] rounded-v-md px-4 py-3 shadow-v-2 dark:shadow-v-dark-2">
            <div className="flex items-center gap-3 mb-2.5">
                <Loader2 className="w-4 h-4 text-link dark:text-[#3291ff] animate-spin flex-shrink-0" />
                <div className="flex-1 min-w-0">
                    <div className="font-semibold text-sm text-link-deep dark:text-[#3291ff] tracking-[-0.28px]">
                        Indexing documents… {status.progress}%
                    </div>
                    <div className="text-[12px] font-mono text-link dark:text-[rgba(50,145,255,0.8)] truncate mt-0.5">
                        {status.current_file || 'Scanning…'}
                    </div>
                </div>
            </div>
            <div className="h-1.5 w-full bg-[rgba(0,112,243,0.15)] dark:bg-[rgba(0,112,243,0.2)] rounded-full overflow-hidden">
                <div
                    className="h-full bg-gradient-to-r from-gradient-develop-start to-gradient-develop-end transition-all duration-300"
                    style={{ width: `${status.progress}%` }}
                />
            </div>
        </div>
    );
}
