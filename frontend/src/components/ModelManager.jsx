import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ModelManager = ({ onSelectModel, activeModel, activeModelPath }) => {
    const [availableModels, setAvailableModels] = useState([]);
    const [localModels, setLocalModels] = useState([]);
    const [downloadStatus, setDownloadStatus] = useState({ downloading: false });
    const [error, setError] = useState(null);
    const [filter, setFilter] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        fetchModels();
        const interval = setInterval(checkDownloadStatus, 2000);
        return () => clearInterval(interval);
    }, []);

    const fetchModels = async () => {
        try {
            const [available, local] = await Promise.all([
                axios.get('http://localhost:8000/api/models/available'),
                axios.get('http://localhost:8000/api/models/local')
            ]);
            setAvailableModels(available.data || []);
            setLocalModels(local.data || []);
        } catch (err) {
            console.error('Failed to fetch models:', err);
        }
    };

    const checkDownloadStatus = async () => {
        try {
            const response = await axios.get('http://localhost:8000/api/models/status');
            setDownloadStatus(response.data);

            if (!response.data.downloading && downloadStatus.downloading) {
                fetchModels();
            }
        } catch (err) {
            // Ignore
        }
    };

    const handleDownload = async (modelId) => {
        setError(null);
        try {
            const response = await axios.post(`http://localhost:8000/api/models/download/${modelId}`);
            if (response.data.status === 'success') {
                setDownloadStatus({ downloading: true, model_id: modelId, progress: 0 });
            }
            if (response.data.message?.includes('Warnings')) {
                setError(response.data.message);
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Download failed');
        }
    };

    const handleDelete = async (modelPath) => {
        if (!confirm('Delete this model?')) return;
        try {
            await axios.delete('http://localhost:8000/api/models', {
                data: { path: modelPath }
            });
            fetchModels();
        } catch (err) {
            setError('Failed to delete');
        }
    };

    const isDownloaded = (modelId) => {
        return localModels.some(m => m.id === modelId || m.filename?.includes(modelId));
    };

    const formatSize = (bytes) => {
        if (!bytes) return '';
        const gb = bytes / (1024 * 1024 * 1024);
        return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
    };

    const normalize = (name) => name?.toLowerCase().replace('.gguf', '') || '';
    const normalizePath = (path) => path?.replace(/\\/g, '/').toLowerCase() || '';

    const getCategoryStyles = (category) => {
        switch (category) {
            case 'small':  return 'bg-green-500/10 text-green-600 dark:text-green-400';
            case 'medium': return 'bg-amber-500/10 text-amber-600 dark:text-amber-400';
            case 'large':  return 'bg-red-500/10 text-red-600 dark:text-red-400';
            default:       return 'bg-slate-500/10 text-slate-600';
        }
    };

    const getCategoryLabel = (category) => {
        switch (category) {
            case 'small':  return 'Optimized';
            case 'medium': return 'Balanced';
            case 'large':  return 'Quality';
            default:       return category;
        }
    };

    const filteredModels = availableModels.filter(model => {
        const matchesFilter = filter === 'all' || model.category === filter;
        const matchesSearch = !searchQuery ||
            model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            model.description.toLowerCase().includes(searchQuery.toLowerCase());
        return matchesFilter && matchesSearch;
    });

    const categories = ['all', 'small', 'medium', 'large'];

    return (
        <div className="space-y-12">
            {error && (
                <div className="flex items-center gap-4 p-5 rounded-3xl bg-red-500/10 text-red-600 dark:text-red-400 text-sm font-bold border border-red-500/20">
                    <span className="material-symbols-outlined">error</span>
                    <span className="flex-1">{error}</span>
                    <button onClick={() => setError(null)} className="material-symbols-outlined text-sm opacity-60 hover:opacity-100 transition-all">close</button>
                </div>
            )}

            {/* Local Models Section */}
            <div className="space-y-6">
                <div className="flex items-center justify-between">
                    <h3 className="text-2xl font-bold font-headline">Downloaded Intelligence</h3>
                    <div className="flex items-center gap-2 px-4 py-2 rounded-2xl bg-primary/5 text-primary text-xs font-black uppercase tracking-widest">
                        <span className="material-symbols-outlined text-sm">memory</span>
                        {localModels.length} Models Ready
                    </div>
                </div>

                {localModels.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {localModels.map((model, idx) => {
                            const isSelected = activeModelPath ? normalizePath(model.path) === normalizePath(activeModelPath) : normalize(model.name) === normalize(activeModel);
                            const isActiveRunning = normalize(model.name) === normalize(activeModel);

                            return (
                                <div key={idx} className={`p-6 rounded-[2.5rem] transition-all border-2 flex flex-col justify-between gap-6 ${isSelected ? 'bg-primary/5 border-primary shadow-xl shadow-primary/5' : 'bg-[#f3f3fd] dark:bg-slate-950/40 border-transparent hover:bg-[#ebebfa]'}`}>
                                    <div className="space-y-4">
                                        <div className="flex items-start justify-between">
                                            <div className="flex items-center gap-3">
                                                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${isSelected ? 'bg-primary text-white' : 'bg-white dark:bg-slate-900 text-primary'}`}>
                                                    <span className="material-symbols-outlined text-2xl">memory</span>
                                                </div>
                                                <div>
                                                    <h4 className="font-bold font-headline text-sm truncate max-w-[150px]">{model.name}</h4>
                                                    <p className="text-[10px] font-black uppercase opacity-40 tracking-widest">{formatSize(model.size)} • Local GGUF</p>
                                                </div>
                                            </div>
                                            {isActiveRunning && (
                                                <span className="px-3 py-1 rounded-full bg-primary/10 text-primary text-[8px] font-black uppercase tracking-widest border border-primary/20">Active</span>
                                            )}
                                        </div>
                                        
                                        <div className="flex items-center gap-4 text-[10px] font-bold opacity-60">
                                            <div className="flex items-center gap-1">
                                                <span className="material-symbols-outlined text-sm">developer_board</span>
                                                {model.ram_required}GB RAM
                                            </div>
                                            <div className="flex items-center gap-1">
                                                <span className="material-symbols-outlined text-sm">settings_b_roll</span>
                                                Q4_K_M
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => onSelectModel && onSelectModel(model)}
                                            className={`flex-1 py-3 rounded-2xl font-bold text-xs transition-all ${isSelected ? 'bg-primary text-white pointer-events-none' : 'bg-white dark:bg-slate-900 text-primary hover:bg-primary hover:text-white shadow-sm'}`}
                                        >
                                            {isSelected ? 'Selected Strategy' : 'Initialize Model'}
                                        </button>
                                        <button
                                            onClick={() => handleDelete(model.path)}
                                            title="Delete Model"
                                            className="w-10 h-10 rounded-2xl bg-white dark:bg-slate-900 text-[#434656] dark:text-slate-400 hover:bg-red-500 hover:text-white flex items-center justify-center transition-all shadow-sm"
                                        >
                                            <span className="material-symbols-outlined text-lg">delete</span>
                                        </button>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <div className="p-12 rounded-[2.5rem] bg-[#f3f3fd] dark:bg-slate-950/40 border-2 border-dashed border-[#d1d1f0] dark:border-slate-800 flex flex-col items-center text-center">
                        <span className="material-symbols-outlined text-5xl text-[#d1d1f0] mb-4">cloud_download</span>
                        <p className="font-bold text-[#434656] opacity-60">No local models found. Download from the registry below.</p>
                    </div>
                )}
            </div>

            {/* Available Registry Section */}
            <div className="space-y-8">
                <div className="flex items-center justify-between">
                    <h3 className="text-2xl font-bold font-headline">Model Registry</h3>
                    <div className="flex gap-2">
                        {categories.map(cat => (
                            <button
                                key={cat}
                                onClick={() => setFilter(cat)}
                                className={`px-4 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${filter === cat ? 'bg-primary text-white shadow-lg shadow-primary/20' : 'bg-[#f3f3fd] dark:bg-slate-950/40 text-[#434656] dark:text-slate-400 hover:bg-[#ebebfa]'}`}
                            >
                                {cat}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Search in Registry */}
                <div className="relative">
                    <span className="material-symbols-outlined absolute left-5 top-1/2 -translate-y-1/2 text-[#434656] opacity-40">search</span>
                    <input
                        type="text"
                        placeholder="Scan for specialized neural weights..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="w-full pl-14 pr-6 py-5 rounded-3xl bg-[#f3f3fd] dark:bg-slate-950/40 border-2 border-transparent focus:border-primary/20 outline-none transition-all font-body text-sm"
                    />
                </div>

                <div className="space-y-4 max-h-[500px] overflow-y-auto custom-scrollbar pr-2">
                    {filteredModels.map((model) => (
                        <div key={model.id} className={`p-6 rounded-[2.5rem] bg-white dark:bg-slate-900 border transition-all ${model.recommended ? 'border-primary/30 shadow-lg shadow-primary/5' : 'border-[#f3f3fd] dark:border-slate-800 shadow-sm'}`}>
                            <div className="flex items-center justify-between gap-6">
                                <div className="flex-1 min-w-0 flex items-center gap-5">
                                    <div className={`w-14 h-14 rounded-2xl flex items-center justify-center shrink-0 ${model.recommended ? 'bg-primary/10 text-primary' : 'bg-[#f3f3fd] dark:bg-slate-800 text-slate-400'}`}>
                                        <span className="material-symbols-outlined text-3xl">{model.recommended ? 'auto_awesome' : 'model_training'}</span>
                                    </div>
                                    <div className="min-w-0">
                                        <div className="flex items-center gap-3 mb-1">
                                            <h4 className="font-bold text-lg font-headline">{model.name}</h4>
                                            <span className={`px-2 py-0.5 rounded-md text-[8px] font-black uppercase tracking-widest ${getCategoryStyles(model.category)}`}>{getCategoryLabel(model.category)}</span>
                                        </div>
                                        <p className="text-xs text-[#434656] dark:text-slate-400 line-clamp-1 mb-2 font-medium">{model.description}</p>
                                        <div className="flex items-center gap-4 text-[10px] font-black uppercase tracking-widest opacity-40">
                                            <span>{model.size}</span>
                                            <span>•</span>
                                            <span className="flex items-center gap-1"><span className="material-symbols-outlined text-xs">memory</span>{model.ram_required}GB RAM</span>
                                            <span>•</span>
                                            <span>{model.quantization}</span>
                                        </div>
                                    </div>
                                </div>

                                <div className="shrink-0">
                                    {isDownloaded(model.id) ? (
                                        <div className="flex items-center gap-2 px-5 py-3 rounded-2xl bg-green-500/10 text-green-600 font-bold text-xs uppercase tracking-widest">
                                            <span className="material-symbols-outlined text-sm">verified</span>
                                            Ready
                                        </div>
                                    ) : downloadStatus.downloading && downloadStatus.model_id === model.id ? (
                                        <div className="flex flex-col items-end gap-2 w-32">
                                            <div className="flex items-center gap-2 text-[10px] font-black text-primary uppercase">
                                                <span>{downloadStatus.progress || 0}%</span>
                                                <span className="material-symbols-outlined animate-spin text-sm">progress_activity</span>
                                            </div>
                                            <div className="w-full h-1.5 bg-primary/10 rounded-full overflow-hidden">
                                                <div className="h-full bg-primary transition-all duration-300" style={{ width: `${downloadStatus.progress || 0}%` }}></div>
                                            </div>
                                        </div>
                                    ) : (
                                        <button
                                            onClick={() => handleDownload(model.id)}
                                            disabled={downloadStatus.downloading}
                                            className="px-6 py-3 rounded-2xl bg-primary text-white font-bold text-xs uppercase tracking-widest hover:shadow-lg hover:shadow-primary/20 transition-all active:scale-95 disabled:opacity-50 flex items-center gap-2"
                                        >
                                            <span className="material-symbols-outlined text-sm">download</span>
                                            Fetch
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                    
                    {filteredModels.length === 0 && (
                        <div className="text-center py-12 space-y-4">
                            <span className="material-symbols-outlined text-5xl opacity-10">search_off</span>
                            <p className="text-sm font-bold opacity-30">No neural weights match the scan query</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ModelManager;
