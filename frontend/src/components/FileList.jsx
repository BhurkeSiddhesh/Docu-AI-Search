import React from 'react';

const FileList = ({ files, onRemove }) => {
    if (!files || files.length === 0) {
        return (
            <div className="text-center py-24 space-y-4 animate-in fade-in duration-500">
                <div className="w-20 h-20 rounded-[2rem] bg-[#f3f3fd] dark:bg-slate-900 mx-auto flex items-center justify-center text-[#d1d1f0]">
                    <span className="material-symbols-outlined text-4xl">inventory_2</span>
                </div>
                <div>
                    <p className="font-bold text-lg text-[#191b22] dark:text-white">Empty Intelligence Base</p>
                    <p className="text-sm opacity-60 font-medium">Index a directory to populate your library.</p>
                </div>
            </div>
        );
    }

    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
    };

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    };

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold font-headline flex items-center gap-3">
                    <span className="material-symbols-outlined text-primary">data_object</span>
                    Neural Assets ({files.length})
                </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {files.map((file) => (
                    <div
                        key={file.id}
                        className="p-6 rounded-[2.5rem] bg-white dark:bg-slate-900 border border-[#f3f3fd] dark:border-slate-800 shadow-sm hover:shadow-xl hover:shadow-primary/5 hover:border-primary/20 transition-all group relative overflow-hidden"
                    >
                        <div className="flex flex-col gap-4 relative z-10">
                            <div className="flex items-start justify-between">
                                <div className="w-12 h-12 rounded-2xl bg-[#f3f3fd] dark:bg-slate-800 flex items-center justify-center text-primary">
                                    <span className="material-symbols-outlined text-2xl">description</span>
                                </div>
                                {onRemove && (
                                    <button
                                        onClick={() => onRemove(file.id)}
                                        className="w-8 h-8 rounded-full bg-[#f3f3fd] dark:bg-slate-800 text-[#434656] dark:text-slate-400 hover:bg-red-500 hover:text-white flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
                                    >
                                        <span className="material-symbols-outlined text-lg">delete</span>
                                    </button>
                                )}
                            </div>

                            <div className="min-w-0">
                                <p className="font-bold text-[#191b22] dark:text-white truncate mb-1" title={file.filename}>
                                    {file.filename}
                                </p>
                                <p className="text-[10px] font-black uppercase opacity-40 tracking-widest truncate" title={file.path}>
                                    {file.path}
                                </p>
                            </div>

                            <div className="flex flex-wrap items-center gap-3">
                                <div className="px-3 py-1.5 rounded-xl bg-[#f3f3fd] dark:bg-slate-800 text-[10px] font-black uppercase tracking-widest opacity-60">
                                    {formatBytes(file.size_bytes)}
                                </div>
                                <div className="px-3 py-1.5 rounded-xl bg-primary/5 text-primary text-[10px] font-black uppercase tracking-widest">
                                    {file.chunk_count} Fragments
                                </div>
                            </div>
                            
                            <div className="flex items-center gap-2 text-[10px] font-bold opacity-40">
                                <span className="material-symbols-outlined text-[12px]">calendar_today</span>
                                {formatDate(file.modified_date)}
                            </div>
                        </div>
                        
                        {/* Decorative background element */}
                        <div className="absolute top-0 right-0 -mr-8 -mt-8 w-24 h-24 bg-primary/5 rounded-full blur-3xl group-hover:bg-primary/10 transition-all duration-700" />
                    </div>
                ))}
            </div>
        </div>
    );
};

export default FileList;
