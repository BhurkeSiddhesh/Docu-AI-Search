import React, { useEffect, useState } from 'react';
import { X, History as HistoryIcon, Trash2, Search } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';
import { formatRelative } from '../lib/format';

export default function HistoryDrawer({ isOpen, onClose, onSelectQuery }) {
    const [items, setItems] = useState([]);
    const [loading, setLoading] = useState(false);
    const toast = useToast();

    useEffect(() => {
        if (!isOpen) return;
        load();
        const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isOpen]);

    const load = async () => {
        setLoading(true);
        try {
            const res = await api.getSearchHistory();
            setItems(res.data || []);
        } catch {
            toast.error('Could not load history');
        } finally {
            setLoading(false);
        }
    };

    const remove = async (id, e) => {
        e.stopPropagation();
        try {
            await api.deleteSearchHistory(id);
            setItems((xs) => xs.filter((x) => x.id !== id));
        } catch {
            toast.error('Could not delete entry');
        }
    };

    const clearAll = async () => {
        if (!confirm('Clear all search history?')) return;
        try {
            await api.clearSearchHistory();
            setItems([]);
            toast.success('History cleared');
        } catch {
            toast.error('Could not clear history');
        }
    };

    if (!isOpen) return null;

    return (
        <>
            <div className="fixed inset-0 bg-[rgba(0,0,0,0.4)] dark:bg-[rgba(0,0,0,0.6)] z-[60] backdrop-blur-sm transition-opacity" onClick={onClose} />
            <aside className="fixed inset-y-0 left-0 z-[70] w-full sm:w-96 bg-canvas dark:bg-[#0a0a0a] border-r border-hairline dark:border-[rgba(255,255,255,0.08)] shadow-v-5 dark:shadow-v-dark-5 flex flex-col animate-fade-in">
                <header className="p-5 border-b border-hairline dark:border-[rgba(255,255,255,0.08)] flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-v-sm bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.06)] flex items-center justify-center border border-hairline dark:border-[rgba(255,255,255,0.1)]">
                            <HistoryIcon className="w-4 h-4 text-ink dark:text-[#ededed]" />
                        </div>
                        <div>
                            <div className="font-semibold text-[15px] text-ink dark:text-[#ededed] tracking-[-0.3px] leading-tight">Search history</div>
                            <div className="text-[11px] font-mono text-mute tracking-[0.05em]">{items.length} entries</div>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-2 rounded-v-sm text-mute hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)] hover:text-ink dark:hover:text-[#ededed] transition" aria-label="Close history">
                        <X className="w-5 h-5" />
                    </button>
                </header>

                <div className="flex-1 overflow-y-auto p-3">
                    {loading ? (
                        <div className="space-y-2">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="card p-3 border-transparent shadow-none">
                                    <div className="h-3 bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.06)] rounded w-2/3 mb-1.5 shimmer" />
                                    <div className="h-2.5 bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.06)] rounded w-1/3 shimmer" />
                                </div>
                            ))}
                        </div>
                    ) : items.length === 0 ? (
                        <div className="text-center py-16">
                            <Search className="w-8 h-8 mx-auto mb-3 text-hairline dark:text-[rgba(255,255,255,0.1)]" />
                            <p className="text-sm font-medium text-body dark:text-[#888]">No history yet</p>
                            <p className="text-xs mt-1 text-mute">Your recent searches will appear here.</p>
                        </div>
                    ) : (
                        <ul className="space-y-1">
                            {items.map((item) => (
                                <li key={item.id} className="group relative">
                                    <button
                                        onClick={() => { onSelectQuery(item.query); onClose(); }}
                                        className="w-full text-left px-3 py-2.5 pr-10 rounded-v-sm hover:bg-canvas-soft dark:hover:bg-[rgba(255,255,255,0.04)] transition"
                                    >
                                        <div className="font-medium text-sm text-ink dark:text-[#ededed] truncate tracking-[-0.28px]">
                                            {item.query}
                                        </div>
                                        <div className="text-[11px] font-mono text-mute mt-0.5">
                                            {formatRelative(item.timestamp)} · {item.result_count} {item.result_count === 1 ? 'result' : 'results'}
                                        </div>
                                    </button>
                                    <button
                                        onClick={(e) => remove(item.id, e)}
                                        className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 focus:opacity-100 transition p-1.5 rounded-v-sm text-mute hover:text-error hover:bg-error-soft dark:hover:bg-[rgba(238,0,0,0.1)]"
                                        aria-label="Delete history entry"
                                    >
                                        <Trash2 className="w-3.5 h-3.5" />
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                {items.length > 0 && (
                    <div className="p-4 border-t border-hairline dark:border-[rgba(255,255,255,0.08)] bg-canvas-soft dark:bg-[#0a0a0a]">
                        <button onClick={clearAll} className="w-full btn-danger">
                            <Trash2 className="w-4 h-4" />
                            Clear all
                        </button>
                    </div>
                )}
            </aside>
        </>
    );
}
