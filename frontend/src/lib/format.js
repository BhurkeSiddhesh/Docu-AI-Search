export function formatBytes(bytes) {
    if (!bytes || bytes < 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function formatRelative(timestamp) {
    if (!timestamp) return '';
    let ts = timestamp;
    // SQLite CURRENT_TIMESTAMP is UTC but serialized without a timezone
    // ("YYYY-MM-DD HH:MM:SS"); parse it as UTC or every entry shows hours old.
    if (typeof ts === 'string' && /^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(ts)) {
        ts = ts.replace(' ', 'T') + 'Z';
    }
    const date = new Date(ts);
    const now = new Date();
    const diff = now - date;
    if (diff < 60_000) return 'just now';
    if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
    if (diff < 604_800_000) return `${Math.floor(diff / 86_400_000)}d ago`;
    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

export function fileExt(name) {
    if (!name) return '';
    const m = name.match(/\.([^.]+)$/);
    return m ? m[1].toUpperCase() : '';
}
