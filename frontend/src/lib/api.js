import axios from 'axios';

const client = axios.create({
    baseURL: '/api',
    timeout: 60000,
});

export const api = {
    // Health
    health: () => client.get('/health'),

    // Config
    getConfig: () => client.get('/config'),
    saveConfig: (data) => client.post('/config', data),

    // Embeddings
    getEmbeddingConfig: () => client.get('/settings/embeddings'),
    saveEmbeddingConfig: (data) => client.post('/settings/embeddings', data),

    // Folders
    browse: () => client.get('/browse'),
    validatePath: (path) => client.post('/validate-path', { path }),
    getFolderHistory: () => client.get('/folders/history'),
    clearFolderHistory: () => client.delete('/folders/history'),
    removeFolderFromHistory: (path) => client.delete('/folders/history/item', { data: { path } }),

    // Indexing
    startIndexing: () => client.post('/index'),
    getIndexStatus: () => client.get('/index/status'),

    // Files
    listFiles: (limit = 100, offset = 0) =>
        client.get(`/files?limit=${limit}&offset=${offset}`),
    previewFile: (path, chars = 2000) =>
        client.get(`/files/preview?path=${encodeURIComponent(path)}&chars=${chars}`),
    openFile: (path) => client.post('/open-file', { path }),

    // Search
    search: (query, opts = {}) =>
        client.post('/search', { query, ...opts }),
    streamAnswer: async (query, context, systemPromptId, onChunk, signal) => {
        const res = await fetch('/api/stream-answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, context, system_prompt_id: systemPromptId }),
            signal,
        });
        if (!res.ok || !res.body) throw new Error('Stream failed');
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                if (chunk) onChunk(chunk);
            }
        } finally {
            reader.cancel();
        }
    },

    // History
    getSearchHistory: () => client.get('/search/history'),
    deleteSearchHistory: (id) => client.delete(`/search/history/${id}`),
    clearSearchHistory: () => client.delete('/search/history'),

    // System prompts
    getSystemPrompts: () => client.get('/system-prompts'),
    createSystemPrompt: (data) => client.post('/system-prompts', data),
    deleteSystemPrompt: (id) => client.delete(`/system-prompts/${id}`),

    // Providers
    listProviders: () => client.get('/providers/list'),
    providerHealth: (data) => client.post('/providers/health', data),
    providerModels: (data) => client.post('/providers/models', data),

    // Models
    listAvailableModels: () => client.get('/models/available'),
    listLocalModels: () => client.get('/models/local'),
    downloadModel: (id) => client.post(`/models/download/${id}`),
    modelDownloadStatus: () => client.get('/models/status'),
    deleteModel: (path) => client.delete('/models/delete', { data: { path } }),

    // Cache
    getCacheStats: () => client.get('/cache/stats'),
    clearCache: () => client.post('/cache/clear'),

    // Benchmarks
    runBenchmarks: () => client.post('/benchmarks/run'),
    benchmarkStatus: () => client.get('/benchmarks/status'),
    benchmarkResults: () => client.get('/benchmarks/results'),
};

export default api;
