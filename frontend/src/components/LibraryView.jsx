import React, { useEffect, useState } from 'react';
import { FileText, FileSpreadsheet, Presentation, FileType, RefreshCw, Database, ChevronLeft, ChevronRight, ExternalLink, Settings as SettingsIcon } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';
import { formatBytes, fileExt } from '../lib/format';

const PAGE_SIZE = 25;

function iconFor(ext) {
    const e = (ext || '').toLowerCase();
    if (e === 'pdf') return FileText;
    if (e === 'docx' || e === 'doc' || e === 'txt') return FileText;
    if (e === 'xlsx' || e === 'xls') return FileSpreadsheet;
    if (e === 'pptx' || e === 'ppt') return Presentation;
    return FileType;
}

export default function LibraryView({ onOpenSettings }) {
    const [files, setFiles] = useState([]);
    const [total, setTotal] = useState(0);
    const [offset, setOffset] = useState(0);
    const [loading, setLoading] = useState(true);
    const [reindexing, setReindexing] = useState(false);
    const toast = useToast();

    const load = async (off = 0) => {
        setLoading(true);
        try {
            const res = await api.listFiles(PAGE_SIZE, off);
            setFiles(res.data.files || []);
            setTotal(res.data.total || 0);
            setOffset(off);
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Failed to load files');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        load(0);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleReindex = async () => {
        setReindexing(true);
        try {
            await api.startIndexing();
            toast.success('Indexing started');
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not start indexing');
        } finally {
            setReindexing(false);
        }
    };

    const handleOpen = async (path) => {
        try {
            await api.openFile(path);
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not open file');
        }
    };

    const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
    const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

    return (
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 lg:py-10">
            <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-slate-900 dark:text-slate-50 mb-1">Library</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                        {total === 0 ? 'No documents indexed yet.' : `${total} indexed document${total === 1 ? '' : 's'}`}
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={onOpenSettings} className="btn-secondary">
                        <SettingsIcon className="w-4 h-4" />
                        Configure folders
                    </button>
                    <button onClick={handleReindex} disabled={reindexing} className="btn-primary">
                        <RefreshCw className={`w-4 h-4 ${reindexing ? 'animate-spin' : ''}`} />
                        Re-index
                    </button>
                </div>
            </div>

            {loading ? (
                <div className="space-y-2">
                    {[1, 2, 3, 4, 5].map((i) => (
                        <div key={i} className="card p-4">
                            <div className="h-4 bg-slate-100 dark:bg-slate-800 rounded w-1/3 mb-2" />
                            <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded w-2/3" />
                        </div>
                    ))}
                </div>
            ) : files.length === 0 ? (
                <div className="card p-10 text-center">
                    <Database className="w-10 h-10 mx-auto mb-3 text-slate-300 dark:text-slate-700" />
                    <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">No indexed documents</p>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
                        Open Settings to add a folder and run an index.
                    </p>
                    <button onClick={onOpenSettings} className="btn-primary">
                        <SettingsIcon className="w-4 h-4" />
                        Open Settings
                    </button>
                </div>
            ) : (
                <>
                    <div className="card overflow-hidden">
                        <div className="divide-y divide-slate-200 dark:divide-slate-800">
                            {files.map((f) => {
                                const ext = fileExt(f.filename).toLowerCase();
                                const Icon = iconFor(ext);
                                return (
                                    <div
                                        key={f.id || f.path}
                                        className="p-4 flex items-center gap-3 hover:bg-slate-50 dark:hover:bg-slate-800/40 transition group"
                                    >
                                        <div className="w-9 h-9 flex-shrink-0 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
                                            <Icon className="w-4 h-4" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <button
                                                onClick={() => f.path && handleOpen(f.path)}
                                                className="font-semibold text-sm text-slate-900 dark:text-slate-50 hover:text-primary text-left truncate block w-full"
                                                title={f.path}
                                            >
                                                {f.filename}
                                            </button>
                                            <div className="text-xs text-slate-500 dark:text-slate-500 truncate font-mono">
                                                {f.path}
                                            </div>
                                        </div>
                                        <div className="hidden sm:flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400 flex-shrink-0">
                                            {ext && (
                                                <span className="chip uppercase text-[10px] tracking-wider">{ext}</span>
                                            )}
                                            {f.size != null && <span>{formatBytes(f.size)}</span>}
                                        </div>
                                        <button
                                            onClick={() => f.path && handleOpen(f.path)}
                                            className="opacity-0 group-hover:opacity-100 transition p-1.5 rounded-md text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700"
                                            title="Open"
                                        >
                                            <ExternalLink className="w-3.5 h-3.5" />
                                        </button>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {totalPages > 1 && (
                        <div className="mt-4 flex items-center justify-between">
                            <div className="text-xs text-slate-500 dark:text-slate-400">
                                Page {currentPage} of {totalPages}
                            </div>
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => load(Math.max(0, offset - PAGE_SIZE))}
                                    disabled={offset === 0 || loading}
                                    className="btn-ghost disabled:opacity-40"
                                >
                                    <ChevronLeft className="w-4 h-4" />
                                    Prev
                                </button>
                                <button
                                    onClick={() => load(offset + PAGE_SIZE)}
                                    disabled={offset + PAGE_SIZE >= total || loading}
                                    className="btn-ghost disabled:opacity-40"
                                >
                                    Next
                                    <ChevronRight className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
