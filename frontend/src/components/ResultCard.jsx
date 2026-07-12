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
        <div className="card p-4 hover:border-hairline-strong dark:hover:border-[rgba(255,255,255,0.2)] transition group">
            <div className="flex items-start gap-3">
                <div className="w-9 h-9 flex-shrink-0 rounded-v-sm bg-canvas-soft dark:bg-[rgba(255,255,255,0.06)] text-ink dark:text-[#ededed] flex items-center justify-center border border-hairline dark:border-[rgba(255,255,255,0.1)]">
                    <Icon className="w-4 h-4" />
                </div>
                <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-3 mb-1">
                        <button
                            onClick={openFile}
                            className="font-semibold text-sm text-ink dark:text-[#ededed] hover:text-link text-left truncate tracking-[-0.28px]"
                            title={result.file_path}
                        >
                            {result.file_name || 'Unknown document'}
                        </button>
                        <div className="flex items-center gap-1 flex-shrink-0">
                            {ext && (
                                <span className="chip text-[10px] font-mono uppercase tracking-[0.05em]">{ext}</span>
                            )}
                            {result.file_path && (
                                <>
                                    <button
                                        onClick={togglePreview}
                                        title="Preview"
                                        className="opacity-0 group-hover:opacity-100 transition p-1.5 rounded-v-sm text-mute hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]"
                                    >
                                        <Eye className="w-3.5 h-3.5" />
                                    </button>
                                    <button
                                        onClick={openFile}
                                        title="Open in system viewer"
                                        className="opacity-0 group-hover:opacity-100 transition p-1.5 rounded-v-sm text-mute hover:bg-canvas-soft-2 dark:hover:bg-[rgba(255,255,255,0.06)]"
                                    >
                                        <ExternalLink className="w-3.5 h-3.5" />
                                    </button>
                                </>
                            )}
                        </div>
                    </div>

                    {result.file_path && (
                        <div className="text-[11px] text-mute truncate mb-2 font-mono">
                            {result.file_path}
                        </div>
                    )}

                    {result.summary && (
                        <p className="text-sm text-body dark:text-[#a1a1a1] leading-relaxed mb-2">
                            {result.summary}
                        </p>
                    )}

                    {/* Skip the raw excerpt when the summary already IS the excerpt
                        (short documents produce identical extractive summaries) */}
                    {result.document && result.document.replace(/\s+/g, ' ').trim() !== (result.summary || '').replace(/\s+/g, ' ').trim() && (
                        <p className="text-xs text-mute line-clamp-3 leading-relaxed">
                            {result.document}
                        </p>
                    )}

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
                            <span className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">
                                Related
                            </span>
                            {result.related_files.slice(0, 3).map((rf, i) => (
                                <button
                                    key={i}
                                    type="button"
                                    onClick={() => api.openFile(rf.path).catch(() => toast.error('Could not open file'))}
                                    title={`${rf.path}\nSimilarity: ${Math.round((rf.similarity || 0) * 100)}%`}
                                    className="chip text-[10px] inline-flex items-center gap-1 hover:text-link hover:border-link/40 transition"
                                >
                                    <Link2 className="w-3 h-3" />
                                    <span className="max-w-[160px] truncate">{rf.filename}</span>
                                </button>
                            ))}
                        </div>
                    )}

                    {showPreview && (
                        <div className="mt-3 pt-3 border-t border-hairline dark:border-[rgba(255,255,255,0.08)] animate-fade-in">
                            <div className="flex items-center justify-between mb-2">
                                <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-mute">Preview</div>
                                <button
                                    onClick={() => setShowPreview(false)}
                                    className="text-mute hover:text-ink dark:hover:text-[#ededed]"
                                >
                                    <X className="w-3.5 h-3.5" />
                                </button>
                            </div>
                            <div className="text-xs text-body dark:text-[#888] whitespace-pre-wrap max-h-64 overflow-y-auto bg-canvas-soft-2 dark:bg-[rgba(255,255,255,0.03)] p-3 rounded-v-sm font-mono leading-relaxed">
                                {loadingPreview ? 'Loading preview...' : previewText || 'No preview available.'}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
