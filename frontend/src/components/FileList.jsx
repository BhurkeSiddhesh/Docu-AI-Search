import React from 'react';
import { File, Trash2, Calendar, HardDrive } from 'lucide-react';
import { cn } from '../lib/utils';

const FileList = ({ files, onRemove }) => {
    if (!files || files.length === 0) {
        return (
            <div className="text-center py-12 text-muted-foreground">
                <HardDrive className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No files indexed yet</p>
                <p className="text-sm mt-2">Index a folder to start searching</p>
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
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    };

    return (
        <div className="space-y-3">
            <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                <HardDrive className="w-5 h-5" />
                Indexed Files ({files.length})
            </h3>

            <div className="grid gap-2 max-h-[400px] overflow-y-auto">
                {files.map((file) => (
                    <div
                        key={file.id}
                        className="p-3 rounded-lg border border-border bg-card hover:bg-accent/50 transition-colors flex items-start justify-between gap-3"
                    >
                        <div className="flex items-start gap-3 flex-1 min-w-0">
                            <File className="w-5 h-5 mt-0.5 text-primary shrink-0" />
                            <div className="min-w-0 flex-1">
                                <p className="font-medium text-foreground truncate" title={file.filename}>
                                    {file.filename}
                                </p>
                                <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground mt-1">
                                    <span className="flex items-center gap-1">
                                        <HardDrive className="w-3 h-3" />
                                        {formatBytes(file.size_bytes)}
                                    </span>
                                    <span className="flex items-center gap-1">
                                        <Calendar className="w-3 h-3" />
                                        {formatDate(file.modified_date)}
                                    </span>
                                    <span className="px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground">
                                        {file.chunk_count} chunks
                                    </span>
                                </div>
                                <p className="text-xs text-muted-foreground mt-1 truncate" title={file.path}>
                                    {file.path}
                                </p>
                            </div>
                        </div>

                        {onRemove && (
                            <button
                                onClick={() => onRemove(file.id)}
                                className="p-2 rounded-md hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors shrink-0"
                                title="Remove from index"
                            >
                                <Trash2 className="w-4 h-4" />
                            </button>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default FileList;
