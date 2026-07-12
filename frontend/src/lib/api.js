import axios from 'axios';

const client = axios.create({
    baseURL: '/api',
    timeout: 60000,
});

// Inject stored Bearer token if AUTH_ENABLED=true on the backend
client.interceptors.request.use((config) => {
    const token = localStorage.getItem('api_token');
    if (token) config.headers['Authorization'] = `Bearer ${token}`;
    return config;
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
    getEmbeddingPresets: () => client.get('/settings/embeddings/presets'),

    // Knowledge graph
    getGraph: () => client.get('/graph'),

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
    streamAnswer: async (query, context, onChunk, signal) => {
        // Raw fetch bypasses the axios interceptor, so attach the token here too
        const headers = { 'Content-Type': 'application/json' };
        const token = localStorage.getItem('api_token');
        if (token) headers['Authorization'] = `Bearer ${token}`;
        const res = await fetch('/api/stream-answer', {
            method: 'POST',
            headers,
            body: JSON.stringify({ query, context }),
            signal,
        });
        if (!res.ok || !res.body) throw new Error('Stream failed');
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    // Flush any buffered multi-byte character split across chunks
                    const tail = decoder.decode();
                    if (tail) onChunk(tail);
                    break;
                }
                // stream: true handles multi-byte characters split across chunks
                const chunk = decoder.decode(value, { stream: true });
                if (chunk) onChunk(chunk);
            }
        } catch (err) {
            if (err.name !== 'AbortError') throw err;
        } finally {
            reader.cancel();
        }
    },

    // Agent
    streamAgentChat: (query, signal) => {
        // Raw fetch (SSE-style stream) — attach the token like streamAnswer does
        const headers = { 'Content-Type': 'application/json' };
        const token = localStorage.getItem('api_token');
        if (token) headers['Authorization'] = `Bearer ${token}`;
        return fetch('/api/agent/chat', {
            method: 'POST',
            headers,
            body: JSON.stringify({ query }),
            signal,
        });
    },

    // Logs (used by lib/logger.js so the auth interceptor applies)
    sendLog: (payload) => client.post('/logs', payload),

    // History
    getSearchHistory: () => client.get('/search/history'),
    deleteSearchHistory: (id) => client.delete(`/search/history/${id}`),
    clearSearchHistory: () => client.delete('/search/history'),


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

    // Auth
    getAuthToken: () => client.get('/auth/token'),
};

export default api;
