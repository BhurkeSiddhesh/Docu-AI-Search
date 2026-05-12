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
            <div className="fixed inset-0 bg-slate-900/40 z-[60]" onClick={onClose} />
            <aside className="fixed inset-y-0 left-0 z-[70] w-full sm:w-96 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 shadow-xl flex flex-col animate-fade-in">
                <header className="p-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
                            <HistoryIcon className="w-4 h-4" />
                        </div>
                        <div>
                            <div className="font-semibold text-slate-900 dark:text-slate-50">Search history</div>
                            <div className="text-xs text-slate-500 dark:text-slate-400">{items.length} entries</div>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800" aria-label="Close history">
                        <X className="w-5 h-5" />
                    </button>
                </header>

                <div className="flex-1 overflow-y-auto p-3">
                    {loading ? (
                        <div className="space-y-2">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="card p-3">
                                    <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded w-2/3 mb-1.5" />
                                    <div className="h-2.5 bg-slate-100 dark:bg-slate-800 rounded w-1/3" />
                                </div>
                            ))}
                        </div>
                    ) : items.length === 0 ? (
                        <div className="text-center py-16 text-slate-500 dark:text-slate-400">
                            <Search className="w-8 h-8 mx-auto mb-2 opacity-30" />
                            <p className="text-sm font-medium">No history yet</p>
                            <p className="text-xs mt-1">Your recent searches will appear here.</p>
                        </div>
                    ) : (
                        <ul className="space-y-1.5">
                            {items.map((item) => (
                                <li key={item.id}>
                                    <button
                                        onClick={() => { onSelectQuery(item.query); onClose(); }}
                                        className="group w-full text-left px-3 py-2.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition flex items-start gap-2"
                                    >
                                        <div className="flex-1 min-w-0">
                                            <div className="font-medium text-sm text-slate-900 dark:text-slate-50 truncate">
                                                {item.query}
                                            </div>
                                            <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                                                {formatRelative(item.timestamp)} · {item.result_count} {item.result_count === 1 ? 'result' : 'results'}
                                            </div>
                                        </div>
                                        <button
                                            onClick={(e) => remove(item.id, e)}
                                            className="opacity-0 group-hover:opacity-100 transition p-1 rounded text-slate-400 hover:text-red-500 flex-shrink-0"
                                            aria-label="Delete history entry"
                                        >
                                            <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                {items.length > 0 && (
                    <div className="p-3 border-t border-slate-200 dark:border-slate-800">
                        <button onClick={clearAll} className="w-full inline-flex items-center justify-center gap-2 text-sm font-medium text-red-600 dark:text-red-400 px-3 py-2 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/40 transition">
                            <Trash2 className="w-4 h-4" />
                            Clear all
                        </button>
                    </div>
                )}
            </aside>
        </>
    );
}
