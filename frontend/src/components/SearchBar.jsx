import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';

const SearchBar = ({ onSearch, isLoading, isAgentMode, onToggleAgent, systemPrompts = [], selectedPromptId, onPromptChange }) => {
    const [query, setQuery] = useState('');
    const [isFocused, setIsFocused] = useState(false);
    const [shortcutHint, setShortcutHint] = useState('Ctrl+K');
    const inputRef = useRef(null);

    useEffect(() => {
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        setShortcutHint(isMac ? '⌘K' : 'Ctrl+K');

        const handleKeyDown = (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                inputRef.current?.focus();
            }
            if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
                e.preventDefault();
                inputRef.current?.focus();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim()) {
            onSearch(query);
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto">
            <form onSubmit={handleSubmit}>
                <motion.div
                    className={`relative flex items-center rounded-2xl md:rounded-3xl transition-all duration-500 bg-white dark:bg-slate-900 ${
                        isFocused ? 'ring-4 ring-primary/10 shadow-2xl' : 'shadow-xl'
                    }`}
                    animate={isFocused ? { scale: 1.01 } : { scale: 1 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                >
                    <div className={`absolute left-4 md:left-6 transition-all duration-300 ${isFocused ? 'text-primary' : 'text-[#434656] dark:text-slate-400'}`}>
                        <span
                            className="material-symbols-outlined text-xl md:text-2xl"
                            style={isFocused ? { fontVariationSettings: "'FILL' 1" } : {}}
                        >
                            search
                        </span>
                    </div>

                    <input
                        ref={inputRef}
                        type="text"
                        aria-label="Search query"
                        className="w-full bg-transparent py-4 md:py-6 pl-12 md:pl-16 pr-28 md:pr-44 text-base md:text-lg font-medium placeholder:text-[#434656]/40 dark:placeholder:text-slate-500 focus:outline-none text-[#191b22] dark:text-white font-body"
                        placeholder={isAgentMode ? 'Ask the AI Researcher...' : 'Search your documents...'}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        disabled={isLoading}
                    />

                    {!isFocused && !query && (
                        <div className="absolute right-28 md:right-36 pointer-events-none hidden md:flex items-center gap-1">
                            <span className="text-[10px] font-bold text-[#434656]/50 dark:text-slate-500 bg-[#f3f3fd] dark:bg-slate-800 rounded-lg px-2 py-1 tracking-widest uppercase">
                                {shortcutHint}
                            </span>
                        </div>
                    )}

                    <div className="absolute right-3 md:right-4 flex items-center gap-1 md:gap-2">
                        {onToggleAgent && (
                            <button
                                type="button"
                                onClick={onToggleAgent}
                                aria-label="Toggle AI Agent Mode"
                                className={`p-2 md:p-3 rounded-xl md:rounded-2xl transition-all duration-300 ${
                                    isAgentMode
                                        ? 'bg-primary/10 text-primary'
                                        : 'hover:bg-surface-container text-[#434656] dark:text-slate-400'
                                }`}
                                title={isAgentMode ? 'Agent Mode Active' : 'Enable Agent Mode'}
                            >
                                <span className="material-symbols-outlined text-lg md:text-xl" style={isAgentMode ? { fontVariationSettings: "'FILL' 1" } : {}}>
                                    smart_toy
                                </span>
                            </button>
                        )}

                        <button
                            type="submit"
                            aria-label="Submit Search"
                            disabled={isLoading || !query.trim()}
                            className="p-2.5 md:p-3 px-4 md:px-6 rounded-xl md:rounded-2xl bg-gradient-to-r from-primary to-primary-container text-white disabled:opacity-40 disabled:grayscale transition-all shadow-lg shadow-primary/20 flex items-center gap-1.5 md:gap-2 active:scale-95"
                        >
                            {isLoading ? (
                                <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
                            ) : (
                                <>
                                    <span className="font-bold text-xs md:text-sm">Analyze</span>
                                    <span className="material-symbols-outlined text-sm hidden sm:inline">bolt</span>
                                </>
                            )}
                        </button>
                    </div>
                </motion.div>
            </form>

            {isAgentMode && (
                <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-4 md:mt-6 flex items-center justify-center gap-3 text-xs font-bold text-primary bg-primary/5 py-3 px-6 rounded-full w-fit mx-auto border border-primary/10"
                >
                    <span className="material-symbols-outlined text-sm">auto_awesome</span>
                    <span className="uppercase tracking-widest">AI Researcher Mode Active</span>
                </motion.div>
            )}

            {!isAgentMode && systemPrompts.length > 0 && (
                <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-5 md:mt-8 flex flex-wrap justify-center gap-3"
                >
                    <div className="flex items-center gap-3 text-[11px] text-[#434656] dark:text-slate-400 bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 rounded-2xl px-4 md:px-6 py-3 shadow-sm hover:shadow-md transition-all">
                        <span className="material-symbols-outlined text-primary text-sm">psychology</span>
                        <span className="font-black uppercase tracking-widest opacity-60 hidden sm:inline">Search Strategy</span>
                        <div className="h-4 w-px bg-[#f3f3fd] dark:bg-slate-800 hidden sm:block" />
                        <select
                            className="bg-transparent border-none focus:ring-0 cursor-pointer font-bold text-[#191b22] dark:text-white p-0 min-w-[100px] md:min-w-[140px] outline-none text-xs md:text-sm"
                            value={selectedPromptId || ''}
                            onChange={(e) => onPromptChange(Number(e.target.value))}
                            disabled={isLoading}
                        >
                            {systemPrompts.map(p => (
                                <option key={p.id} value={p.id} className="bg-white dark:bg-slate-900">{p.name}</option>
                            ))}
                        </select>
                    </div>
                </motion.div>
            )}
        </div>
    );
};

export default SearchBar;
