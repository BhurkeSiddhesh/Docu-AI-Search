import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';

const getFileIcon = (fileName) => {
    const ext = fileName?.split('.').pop()?.toLowerCase();
    switch (ext) {
        case 'pdf': return 'picture_as_pdf';
        case 'docx':
        case 'doc': return 'description';
        case 'xlsx':
        case 'xls': return 'table_chart';
        case 'pptx':
        case 'ppt': return 'slideshow';
        case 'png':
        case 'jpg':
        case 'jpeg': return 'image';
        default: return 'draft';
    }
};

const handleOpenFile = async (filePath) => {
    try {
        await axios.post('/api/open-file', { path: filePath });
    } catch (error) {
        console.error('Failed to open file:', error);
    }
};

const SearchResults = ({ results, aiAnswer }) => {
    if (!results.length && !aiAnswer) return null;

    const resultsList = useMemo(() => {
        return results.map((result, index) => (
            <motion.div
                key={result.faiss_idx !== undefined ? result.faiss_idx : index}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="bg-white dark:bg-slate-900 rounded-2xl md:rounded-[2rem] p-4 md:p-6 hover:shadow-2xl hover:shadow-primary/10 transition-all border border-[#f3f3fd] dark:border-slate-800 group cursor-pointer mb-4 md:mb-6 result-card"
                onClick={() => result.file_path && handleOpenFile(result.file_path)}
                role="listitem"
                tabIndex={0}
                aria-label={`Open ${result.file_name}`}
                onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        result.file_path && handleOpenFile(result.file_path);
                    }
                }}
            >
                <div className="flex items-center justify-between mb-3 md:mb-4">
                    <div className="flex items-center gap-2 md:gap-3 min-w-0">
                        <div className="w-10 h-10 md:w-12 md:h-12 flex-shrink-0 rounded-xl md:rounded-2xl bg-[#f3f3fd] dark:bg-slate-800 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-white transition-all">
                            <span className="material-symbols-outlined text-xl md:text-2xl">{getFileIcon(result.file_name)}</span>
                        </div>
                        <div className="min-w-0">
                            <h4 className="font-bold text-[#191b22] dark:text-white truncate text-sm md:text-base max-w-[160px] sm:max-w-[280px] md:max-w-[400px]">
                                {result.file_name}
                            </h4>
                            <p className="text-[9px] md:text-[10px] font-black uppercase tracking-widest text-[#434656] dark:text-slate-500 opacity-60">
                                {result.file_name?.split('.').pop()?.toUpperCase()} Document
                            </p>
                        </div>
                    </div>
                    <button
                        title="Open file externally"
                        className="w-9 h-9 md:w-10 md:h-10 flex-shrink-0 rounded-full hover:bg-[#f3f3fd] dark:hover:bg-slate-800 flex items-center justify-center transition-all"
                    >
                        <span className="material-symbols-outlined text-lg md:text-xl">open_in_new</span>
                    </button>
                </div>

                <div className="bg-[#f3f3fd]/50 dark:bg-slate-800/50 rounded-xl md:rounded-2xl p-3 md:p-4 mt-2">
                    <p className="text-xs md:text-sm text-[#434656] dark:text-slate-400 leading-relaxed line-clamp-3 font-medium">
                        {result.document}
                    </p>
                </div>
            </motion.div>
        ));
    }, [results]);

    return (
        <div className="w-full mt-8 md:mt-12 pb-20 md:pb-24">
            <p className="sr-only" aria-live="polite" aria-atomic="true">
                {results.length} search result{results.length !== 1 ? 's' : ''} found
            </p>

            {/* On mobile: AI panel first, then results. On large: side-by-side */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 md:gap-8 items-start">
                {/* AI Summary — mobile: top, desktop: right sticky panel */}
                {aiAnswer && (
                    <motion.aside
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="lg:col-span-4 lg:col-start-9 lg:row-start-1 lg:sticky lg:top-28 order-first lg:order-none"
                    >
                        <div className="bg-white dark:bg-slate-900 rounded-2xl md:rounded-[2.5rem] p-5 md:p-8 shadow-2xl border border-primary/5 relative overflow-hidden">
                            <div className="absolute top-0 right-0 w-24 h-24 bg-primary/5 rounded-bl-[100px] -mr-8 -mt-8 hidden md:block" />

                            <div className="flex items-center gap-3 md:gap-4 mb-5 md:mb-8">
                                <div className="w-10 h-10 md:w-12 md:h-12 rounded-xl md:rounded-2xl bg-primary text-white flex items-center justify-center shadow-lg shadow-primary/20 flex-shrink-0">
                                    <span className="material-symbols-outlined fill-current text-lg md:text-xl">auto_awesome</span>
                                </div>
                                <div>
                                    <h3 className="text-base md:text-lg font-bold text-[#191b22] dark:text-white">AI Synthesis</h3>
                                    <p className="text-[9px] md:text-[10px] font-black uppercase tracking-widest text-primary">Contextual Analysis</p>
                                </div>
                            </div>

                            <div className="relative z-10">
                                <div className="text-[#434656] dark:text-slate-300 text-xs md:text-sm leading-relaxed font-medium space-y-4">
                                    {aiAnswer === "Thinking..." || aiAnswer === "Generating summary..." ? (
                                        <div className="flex items-center gap-3">
                                            <span className="material-symbols-outlined animate-spin text-primary">progress_activity</span>
                                            <span>Processing knowledge graph...</span>
                                        </div>
                                    ) : (
                                        <p className="whitespace-pre-wrap">{aiAnswer}</p>
                                    )}
                                </div>
                            </div>

                            <div className="mt-6 md:mt-10 pt-4 md:pt-6 border-t border-[#f3f3fd] dark:border-slate-800 flex items-center justify-between">
                                <div className="flex -space-x-2">
                                    {[1, 2, 3].map(i => (
                                        <div key={i} className="w-5 h-5 md:w-6 md:h-6 rounded-full bg-surface-container border-2 border-white dark:border-slate-900 flex items-center justify-center">
                                            <span className="material-symbols-outlined text-[9px] md:text-[10px] text-primary">verified</span>
                                        </div>
                                    ))}
                                </div>
                                <button className="text-[10px] font-black uppercase tracking-widest text-primary hover:underline">Copy Insight</button>
                            </div>
                        </div>

                        <div className="mt-4 md:mt-6 bg-gradient-to-br from-primary to-primary-container rounded-2xl md:rounded-[2rem] p-4 md:p-6 text-white shadow-xl shadow-primary/10">
                            <h4 className="font-bold text-sm mb-1">Deep Search Active</h4>
                            <p className="text-[10px] opacity-80 font-medium">Results are cross-referenced with your local library using neural embeddings.</p>
                        </div>
                    </motion.aside>
                )}

                {/* Main Results Column */}
                <div className={`${aiAnswer ? 'lg:col-span-8 lg:col-start-1 lg:row-start-1' : 'lg:col-span-12'}`}>
                    <div className="flex items-center justify-between mb-5 md:mb-8">
                        <h3 className="text-base md:text-xl font-black uppercase tracking-widest text-[#191b22] dark:text-white flex items-center gap-2 md:gap-3">
                            <span className="w-1.5 md:w-2 h-6 md:h-8 bg-primary rounded-full" />
                            Found Knowledge
                        </h3>
                        <p className="text-xs font-bold text-[#434656]/60">{results.length} relevant chunks</p>
                    </div>
                    <div role="list" aria-label="Search results">
                        {resultsList}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SearchResults;
