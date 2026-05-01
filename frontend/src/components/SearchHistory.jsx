import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';

const API = '/api';

const SearchHistory = ({ onSelectQuery, isOpen, onClose }) => {
    const [history, setHistory] = useState([]);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (isOpen) {
            fetchHistory();
        }
    }, [isOpen]);

    const fetchHistory = async () => {
        setIsLoading(true);
        try {
            const response = await axios.get(`${API}/search/history`);
            setHistory(response.data || []);
        } catch (error) {
            console.error('Failed to fetch history:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const deleteHistoryItem = async (id, e) => {
        e.stopPropagation();
        try {
            await axios.delete(`${API}/search/history/${id}`);
            setHistory(history.filter(item => item.id !== id));
        } catch (error) {
            console.error('Failed to delete:', error);
        }
    };

    const formatTime = (timestamp) => {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        if (diff < 60000) return 'now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-slate-950/20 backdrop-blur-sm z-[60]"
                    />
                    <motion.div
                        initial={{ x: '-100%' }}
                        animate={{ x: 0 }}
                        exit={{ x: '-100%' }}
                        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                        className="fixed left-0 md:left-72 top-0 bottom-0 w-full sm:w-96 bg-white/95 dark:bg-slate-900/95 backdrop-blur-2xl z-[70] shadow-2xl border-r border-[#f3f3fd] dark:border-slate-800 flex flex-col"
                    >
                        <div className="p-5 md:p-8 border-b border-[#f3f3fd] dark:border-slate-800 flex items-center justify-between">
                            <div className="flex items-center gap-3 md:gap-4">
                                <div className="w-9 h-9 md:w-10 md:h-10 rounded-xl md:rounded-2xl bg-primary/5 text-primary flex items-center justify-center">
                                    <span className="material-symbols-outlined text-xl md:text-2xl">history</span>
                                </div>
                                <div>
                                    <h2 className="text-lg md:text-xl font-bold font-headline">History</h2>
                                    <p className="text-[9px] md:text-[10px] font-black opacity-40 uppercase tracking-widest">Recent Neural Scans</p>
                                </div>
                            </div>
                            <button onClick={onClose} className="w-9 h-9 md:w-10 md:h-10 rounded-full hover:bg-[#f3f3fd] dark:hover:bg-slate-800 flex items-center justify-center transition-all">
                                <span className="material-symbols-outlined">close</span>
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-3 md:space-y-4 custom-scrollbar">
                            {isLoading ? (
                                <div className="flex flex-col items-center justify-center py-20 gap-4">
                                    <span className="material-symbols-outlined animate-spin text-primary text-4xl">progress_activity</span>
                                    <p className="text-xs font-black uppercase tracking-widest opacity-40">Accessing Archives...</p>
                                </div>
                            ) : history.length === 0 ? (
                                <div className="flex flex-col items-center justify-center py-32 text-center opacity-40 space-y-4">
                                    <span className="material-symbols-outlined text-5xl md:text-6xl">search_off</span>
                                    <p className="text-sm font-bold">No historical data available</p>
                                </div>
                            ) : (
                                history.map((item) => (
                                    <div
                                        key={item.id}
                                        onClick={() => { onSelectQuery(item.query); onClose(); }}
                                        className="group relative p-4 md:p-5 rounded-2xl md:rounded-3xl bg-[#f3f3fd] dark:bg-slate-800/40 hover:bg-white dark:hover:bg-slate-800 transition-all cursor-pointer border border-transparent hover:border-primary/20 hover:shadow-xl hover:shadow-primary/5"
                                    >
                                        <div className="flex-1 min-w-0 pr-8">
                                            <p className="text-sm font-bold text-[#191b22] dark:text-white truncate mb-1 group-hover:text-primary transition-colors">{item.query}</p>
                                            <div className="flex items-center gap-2 md:gap-3 text-[10px] font-black uppercase tracking-widest opacity-40">
                                                <span className="flex items-center gap-1">
                                                    <span className="material-symbols-outlined text-[12px]">schedule</span>
                                                    {formatTime(item.timestamp)}
                                                </span>
                                                <span>•</span>
                                                <span>{item.result_count} Neural Hits</span>
                                            </div>
                                        </div>
                                        <button
                                            onClick={(e) => deleteHistoryItem(item.id, e)}
                                            aria-label="Delete history item"
                                            className="absolute right-3 top-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white dark:bg-slate-900 shadow-sm opacity-0 group-hover:opacity-100 hover:bg-red-500 hover:text-white flex items-center justify-center transition-all"
                                        >
                                            <span className="material-symbols-outlined text-lg">delete</span>
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>

                        <div className="p-4 md:p-6 bg-[#f3f3fd] dark:bg-slate-950/40 border-t border-[#f3f3fd] dark:border-slate-800">
                            <button
                                onClick={async () => {
                                    if (confirm('Clear history?')) {
                                        await axios.delete(`${API}/search/history`);
                                        fetchHistory();
                                    }
                                }}
                                className="w-full py-3 md:py-4 rounded-2xl bg-white dark:bg-slate-900 font-bold text-[10px] uppercase tracking-widest text-red-500 hover:bg-red-500 hover:text-white transition-all shadow-sm flex items-center justify-center gap-2"
                            >
                                <span className="material-symbols-outlined text-sm">delete_sweep</span>
                                Wipe History Archive
                            </button>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};

export default SearchHistory;
