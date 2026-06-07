/**
 * Tests for frontend/src/lib/api.js
 *
 * Verifies that every method on the `api` object makes the correct
 * HTTP call (method + path) via the axios client.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';

vi.mock('axios', () => {
    const mockClient = {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
    };
    return {
        default: {
            ...mockClient,
            create: vi.fn(() => mockClient),
        },
    };
});

import { api } from '../lib/api';

const mockClient = axios.create();

beforeEach(() => {
    vi.clearAllMocks();
    mockClient.get.mockResolvedValue({ data: {} });
    mockClient.post.mockResolvedValue({ data: {} });
    mockClient.delete.mockResolvedValue({ data: {} });
});

// ── Shape of the api export ────────────────────────────────────────────────────

describe('api export shape', () => {
    const expectedMethods = [
        'health',
        'getConfig', 'saveConfig',
        'getEmbeddingConfig', 'saveEmbeddingConfig',
        'browse', 'validatePath', 'getFolderHistory', 'clearFolderHistory', 'removeFolderFromHistory',
        'startIndexing', 'getIndexStatus',
        'listFiles', 'previewFile', 'openFile',
        'search', 'streamAnswer',
        'getSearchHistory', 'deleteSearchHistory', 'clearSearchHistory',
        'getSystemPrompts', 'createSystemPrompt', 'deleteSystemPrompt',
        'listProviders', 'providerHealth', 'providerModels',
        'listAvailableModels', 'listLocalModels', 'downloadModel', 'modelDownloadStatus', 'deleteModel',
        'getCacheStats', 'clearCache',
        'runBenchmarks', 'benchmarkStatus', 'benchmarkResults',
    ];

    it.each(expectedMethods)('exports method: %s', (method) => {
        expect(typeof api[method]).toBe('function');
    });
});

// ── GET endpoints ─────────────────────────────────────────────────────────────

describe('GET endpoints', () => {
    it('health() calls GET /health', () => {
        api.health();
        expect(mockClient.get).toHaveBeenCalledWith('/health');
    });

    it('getConfig() calls GET /config', () => {
        api.getConfig();
        expect(mockClient.get).toHaveBeenCalledWith('/config');
    });

    it('getEmbeddingConfig() calls GET /settings/embeddings', () => {
        api.getEmbeddingConfig();
        expect(mockClient.get).toHaveBeenCalledWith('/settings/embeddings');
    });

    it('browse() calls GET /browse', () => {
        api.browse();
        expect(mockClient.get).toHaveBeenCalledWith('/browse');
    });

    it('getFolderHistory() calls GET /folders/history', () => {
        api.getFolderHistory();
        expect(mockClient.get).toHaveBeenCalledWith('/folders/history');
    });

    it('getIndexStatus() calls GET /index/status', () => {
        api.getIndexStatus();
        expect(mockClient.get).toHaveBeenCalledWith('/index/status');
    });

    it('listFiles() calls GET /files with default limit and offset', () => {
        api.listFiles();
        expect(mockClient.get).toHaveBeenCalledWith('/files?limit=100&offset=0');
    });

    it('listFiles(50, 10) calls GET /files with correct params', () => {
        api.listFiles(50, 10);
        expect(mockClient.get).toHaveBeenCalledWith('/files?limit=50&offset=10');
    });

    it('previewFile(path) calls GET with encoded path', () => {
        api.previewFile('/some/path/file.pdf');
        expect(mockClient.get).toHaveBeenCalledWith(
            expect.stringContaining('/files/preview?path=')
        );
        expect(mockClient.get).toHaveBeenCalledWith(
            expect.stringContaining(encodeURIComponent('/some/path/file.pdf'))
        );
    });

    it('getSearchHistory() calls GET /search/history', () => {
        api.getSearchHistory();
        expect(mockClient.get).toHaveBeenCalledWith('/search/history');
    });

    it('getSystemPrompts() calls GET /system-prompts', () => {
        api.getSystemPrompts();
        expect(mockClient.get).toHaveBeenCalledWith('/system-prompts');
    });

    it('listProviders() calls GET /providers/list', () => {
        api.listProviders();
        expect(mockClient.get).toHaveBeenCalledWith('/providers/list');
    });

    it('listAvailableModels() calls GET /models/available', () => {
        api.listAvailableModels();
        expect(mockClient.get).toHaveBeenCalledWith('/models/available');
    });

    it('listLocalModels() calls GET /models/local', () => {
        api.listLocalModels();
        expect(mockClient.get).toHaveBeenCalledWith('/models/local');
    });

    it('modelDownloadStatus() calls GET /models/status', () => {
        api.modelDownloadStatus();
        expect(mockClient.get).toHaveBeenCalledWith('/models/status');
    });

    it('getCacheStats() calls GET /cache/stats', () => {
        api.getCacheStats();
        expect(mockClient.get).toHaveBeenCalledWith('/cache/stats');
    });

    it('benchmarkStatus() calls GET /benchmarks/status', () => {
        api.benchmarkStatus();
        expect(mockClient.get).toHaveBeenCalledWith('/benchmarks/status');
    });

    it('benchmarkResults() calls GET /benchmarks/results', () => {
        api.benchmarkResults();
        expect(mockClient.get).toHaveBeenCalledWith('/benchmarks/results');
    });
});

// ── POST endpoints ────────────────────────────────────────────────────────────

describe('POST endpoints', () => {
    it('saveConfig(data) calls POST /config with data', () => {
        const data = { provider: 'openai' };
        api.saveConfig(data);
        expect(mockClient.post).toHaveBeenCalledWith('/config', data);
    });

    it('saveEmbeddingConfig(data) calls POST /settings/embeddings', () => {
        const data = { provider_type: 'local' };
        api.saveEmbeddingConfig(data);
        expect(mockClient.post).toHaveBeenCalledWith('/settings/embeddings', data);
    });

    it('validatePath(path) calls POST /validate-path with path object', () => {
        api.validatePath('/home/user/docs');
        expect(mockClient.post).toHaveBeenCalledWith('/validate-path', { path: '/home/user/docs' });
    });

    it('startIndexing() calls POST /index', () => {
        api.startIndexing();
        expect(mockClient.post).toHaveBeenCalledWith('/index');
    });

    it('openFile(path) calls POST /open-file with path', () => {
        api.openFile('/some/file.pdf');
        expect(mockClient.post).toHaveBeenCalledWith('/open-file', { path: '/some/file.pdf' });
    });

    it('search(query, opts) calls POST /search with merged payload', () => {
        api.search('machine learning', { top_k: 5 });
        expect(mockClient.post).toHaveBeenCalledWith('/search', {
            query: 'machine learning',
            top_k: 5,
        });
    });

    it('search(query) with no opts sends just the query', () => {
        api.search('revenue');
        expect(mockClient.post).toHaveBeenCalledWith('/search', { query: 'revenue' });
    });

    it('createSystemPrompt(data) calls POST /system-prompts', () => {
        const data = { name: 'My Prompt', content: 'Answer concisely.' };
        api.createSystemPrompt(data);
        expect(mockClient.post).toHaveBeenCalledWith('/system-prompts', data);
    });

    it('providerHealth(data) calls POST /providers/health', () => {
        api.providerHealth({ provider: 'openai' });
        expect(mockClient.post).toHaveBeenCalledWith('/providers/health', { provider: 'openai' });
    });

    it('providerModels(data) calls POST /providers/models', () => {
        api.providerModels({ provider: 'gemini' });
        expect(mockClient.post).toHaveBeenCalledWith('/providers/models', { provider: 'gemini' });
    });

    it('downloadModel(id) calls POST /models/download/:id', () => {
        api.downloadModel('llama-3-8b');
        expect(mockClient.post).toHaveBeenCalledWith('/models/download/llama-3-8b');
    });

    it('clearCache() calls POST /cache/clear', () => {
        api.clearCache();
        expect(mockClient.post).toHaveBeenCalledWith('/cache/clear');
    });

    it('runBenchmarks() calls POST /benchmarks/run', () => {
        api.runBenchmarks();
        expect(mockClient.post).toHaveBeenCalledWith('/benchmarks/run');
    });
});

// ── DELETE endpoints ──────────────────────────────────────────────────────────

describe('DELETE endpoints', () => {
    it('clearFolderHistory() calls DELETE /folders/history', () => {
        api.clearFolderHistory();
        expect(mockClient.delete).toHaveBeenCalledWith('/folders/history');
    });

    it('removeFolderFromHistory(path) calls DELETE /folders/history/item with data', () => {
        api.removeFolderFromHistory('/old/path');
        expect(mockClient.delete).toHaveBeenCalledWith(
            '/folders/history/item',
            { data: { path: '/old/path' } }
        );
    });

    it('deleteSearchHistory(id) calls DELETE /search/history/:id', () => {
        api.deleteSearchHistory(42);
        expect(mockClient.delete).toHaveBeenCalledWith('/search/history/42');
    });

    it('clearSearchHistory() calls DELETE /search/history', () => {
        api.clearSearchHistory();
        expect(mockClient.delete).toHaveBeenCalledWith('/search/history');
    });

    it('deleteSystemPrompt(id) calls DELETE /system-prompts/:id', () => {
        api.deleteSystemPrompt(7);
        expect(mockClient.delete).toHaveBeenCalledWith('/system-prompts/7');
    });

    it('deleteModel(path) calls DELETE /models/delete with data', () => {
        api.deleteModel('/models/llama.gguf');
        expect(mockClient.delete).toHaveBeenCalledWith(
            '/models/delete',
            { data: { path: '/models/llama.gguf' } }
        );
    });
});

// ── streamAnswer ──────────────────────────────────────────────────────────────

describe('streamAnswer', () => {
    it('calls fetch at /api/stream-answer with POST method', async () => {
        const chunks = [];
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
            start(controller) {
                controller.enqueue(encoder.encode('chunk1'));
                controller.close();
            }
        });

        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            body: stream,
        });

        await api.streamAnswer('query', 'context', null, (chunk) => chunks.push(chunk));

        expect(global.fetch).toHaveBeenCalledWith(
            '/api/stream-answer',
            expect.objectContaining({ method: 'POST' })
        );
    });

    it('passes query, context, and system_prompt_id in the request body', async () => {
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
            start(controller) { controller.close(); }
        });

        global.fetch = vi.fn().mockResolvedValue({ ok: true, body: stream });

        await api.streamAnswer('my query', 'some context', 3, () => {});

        const callArgs = global.fetch.mock.calls[0];
        const body = JSON.parse(callArgs[1].body);
        expect(body.query).toBe('my query');
        expect(body.context).toBe('some context');
        expect(body.system_prompt_id).toBe(3);
    });

    it('throws when the response is not ok', async () => {
        global.fetch = vi.fn().mockResolvedValue({ ok: false, body: null });
        await expect(api.streamAnswer('q', 'c', null, () => {})).rejects.toThrow('Stream failed');
    });

    it('calls onChunk for each streamed chunk', async () => {
        const chunks = [];
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
            start(controller) {
                controller.enqueue(encoder.encode('part1'));
                controller.enqueue(encoder.encode('part2'));
                controller.close();
            }
        });

        global.fetch = vi.fn().mockResolvedValue({ ok: true, body: stream });

        await api.streamAnswer('q', 'c', null, (chunk) => chunks.push(chunk));

        expect(chunks.length).toBeGreaterThan(0);
        expect(chunks.join('')).toContain('part1');
    });
});
