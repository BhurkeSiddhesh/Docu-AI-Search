/**
 * Logger Utility Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import axios from 'axios';

vi.mock('axios');

// Import after mock is set up
import logger from '../lib/logger';

beforeEach(() => {
    vi.clearAllMocks();
    axios.post.mockResolvedValue({ status: 200 });
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
    vi.restoreAllMocks();
});

describe('Logger Utility', () => {
    it('exports an object with log, info, warn, error, and init methods', () => {
        expect(typeof logger.log).toBe('function');
        expect(typeof logger.info).toBe('function');
        expect(typeof logger.warn).toBe('function');
        expect(typeof logger.error).toBe('function');
        expect(typeof logger.init).toBe('function');
    });

    it('logger.info posts to the backend with level "info"', async () => {
        await logger.info('hello world');
        expect(axios.post).toHaveBeenCalledWith(
            expect.stringContaining('/api/logs'),
            expect.objectContaining({ level: 'info', message: 'hello world', source: 'Frontend' })
        );
    });

    it('logger.warn posts to the backend with level "warn"', async () => {
        await logger.warn('something suspicious');
        expect(axios.post).toHaveBeenCalledWith(
            expect.any(String),
            expect.objectContaining({ level: 'warn', message: 'something suspicious' })
        );
    });

    it('logger.error posts with level "error" and optional stack', async () => {
        const stack = 'Error at line 42';
        await logger.error('crash', stack);
        expect(axios.post).toHaveBeenCalledWith(
            expect.any(String),
            expect.objectContaining({ level: 'error', message: 'crash', stack })
        );
    });

    it('logger.error works without a stack argument', async () => {
        await logger.error('no stack error');
        expect(axios.post).toHaveBeenCalledWith(
            expect.any(String),
            expect.objectContaining({ level: 'error', message: 'no stack error', stack: null })
        );
    });

    it('serializes non-string messages to JSON', async () => {
        await logger.log('info', { key: 'value' });
        expect(axios.post).toHaveBeenCalledWith(
            expect.any(String),
            expect.objectContaining({ message: '{"key":"value"}' })
        );
    });

    it('does not throw when the backend post fails', async () => {
        axios.post.mockRejectedValueOnce(new Error('Network down'));
        await expect(logger.info('safe')).resolves.not.toThrow();
    });

    it('always logs to console regardless of backend success', async () => {
        await logger.info('console check');
        expect(console.log).toHaveBeenCalled();
    });

    it('logger.init sets window.onerror and window.onunhandledrejection', () => {
        logger.init();
        expect(typeof window.onerror).toBe('function');
        expect(typeof window.onunhandledrejection).toBe('function');
    });
});
