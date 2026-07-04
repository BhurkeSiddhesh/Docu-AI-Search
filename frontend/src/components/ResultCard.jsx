import React, { useState } from 'react';
import { FileText, FileSpreadsheet, Presentation, FileType, ExternalLink, Eye, X, Link2 } from 'lucide-react';
import api from '../lib/api';
import { useToast } from './Toast';
import { fileExt } from '../lib/format';

function iconFor(ext) {
    const e = (ext || '').toLowerCase();
    if (e === 'pdf') return FileText;
    if (e === 'docx' || e === 'doc' || e === 'txt') return FileText;
    if (e === 'xlsx' || e === 'xls') return FileSpreadsheet;
    if (e === 'pptx' || e === 'ppt') return Presentation;
    return FileType;
}

export default function ResultCard({ result }) {
    const toast = useToast();
    const [showPreview, setShowPreview] = useState(false);
    const [previewText, setPreviewText] = useState('');
    const [loadingPreview, setLoadingPreview] = useState(false);

    const ext = fileExt(result.file_name).toLowerCase();
    const Icon = iconFor(ext);

    const openFile = async () => {
        if (!result.file_path) {
            toast.error('No file path available');
            return;
        }
        try {
            await api.openFile(result.file_path);
        } catch (e) {
            toast.error(e.response?.data?.detail || 'Could not open file');
        }
    };

    const togglePreview = async () => {
        if (showPreview) {
            setShowPreview(false);
            return;
        }
        if (!result.file_path) return;
        setShowPreview(true);
        if (!previewText) {
            setLoadingPreview(true);
            try {
                const res = await api.previewFile(result.file_path, 2000);
                setPreviewText(res.data.preview || '');
            } catch (e) {
                setPreviewText('Could not load preview.');
            } finally {
                setLoadingPreview(false);
            }
        }
    };

    return (
        <div className="card p-4 hover:border-slate-300 dark:hover:border-slate-700 transition group">
            <div className="flex items-start gap-3">
                <div className="w-9 h-9 flex-shrink-0 rounded-lg bg-primary/10 text-primary flex items-center justify-center">
                    <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-3 mb-1">
                        <button
                            onClick={openFile}
                            className="font-semibold text-sm text-slate-900 dark:text-slate-50 hover:text-primary text-left truncate"
                            title={result.file_path}
                        >
                            {result.file_name || 'Unknown document'}
                        </button>
                        <div className="flex items-center gap-1 flex-shrink-0">
                            {ext && (
                                <span className="chip uppercase text-[10px] tracking-wider">{ext}</span>
                            )}
                            {result.file_path && (
                                <>
                                    <button
                                        onClick={togglePreview}
                                        title="Preview"
                                        className="opacity-0 group-hover:opacity-100 transition p-1.5 rounded-md text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                                    >
                                        <Eye className="w-3.5 h-3.5" />
                                    </button>
                                    <button
                                        onClick={openFile}
                                        title="Open in system viewer"
                                        className="opacity-0 group-hover:opacity-100 transition p-1.5 rounded-md text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                                    >
                                        <ExternalLink className="w-3.5 h-3.5" />
                                    </button>
                                </>
                            )}
                        </div>
                    </div>

                    {result.file_path && (
                        <div className="text-[11px] text-slate-500 dark:text-slate-500 truncate mb-2 font-mono">
                            {result.file_path}
                        </div>
                    )}

                    {result.summary && (
                        <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed mb-2">
                            {result.summary}
                        </p>
                    )}

                    <p className="text-xs text-slate-500 dark:text-slate-400 line-clamp-3 leading-relaxed">
                        {result.document}
                    </p>

                    {result.tags?.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mt-3">
                            {result.tags.slice(0, 6).map((tag, i) => (
                                <span key={i} className="chip text-[10px]">
                                    {tag}
                                </span>
                            ))}
                        </div>
                    )}

                    {result.related_files?.length > 0 && (
                        <div className="flex flex-wrap items-center gap-1.5 mt-3">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                                Related
                            </span>
                            {result.related_files.slice(0, 3).map((rf, i) => (
                                <button
                                    key={i}
                                    type="button"
                                    onClick={() => api.openFile(rf.path).catch(() => toast.error('Could not open file'))}
                                    title={`${rf.path}\nSimilarity: ${Math.round((rf.similarity || 0) * 100)}%`}
                                    className="chip text-[10px] inline-flex items-center gap-1 hover:text-primary hover:border-primary/40 transition"
                                >
                                    <Link2 className="w-3 h-3" />
                                    <span className="max-w-[160px] truncate">{rf.filename}</span>
                                </button>
                            ))}
                        </div>
                    )}

                    {showPreview && (
                        <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-800 animate-fade-in">
                            <div className="flex items-center justify-between mb-2">
                                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Preview</div>
                                <button
                                    onClick={() => setShowPreview(false)}
                                    className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                                >
                                    <X className="w-3.5 h-3.5" />
                                </button>
                            </div>
                            <div className="text-xs text-slate-600 dark:text-slate-400 whitespace-pre-wrap max-h-64 overflow-y-auto bg-slate-50 dark:bg-slate-950 p-3 rounded-lg font-mono leading-relaxed">
                                {loadingPreview ? 'Loading preview…' : previewText || 'No preview available.'}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
