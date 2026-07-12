/**
 * Logger Utility Tests
 *
 * logger routes uploads through the shared API client (api.sendLog) so the
 * auth interceptor and relative /api base apply — the mock mirrors that.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('../lib/api', () => ({
    default: { sendLog: vi.fn() },
}));

// Import after mock is set up
import api from '../lib/api';
import logger from '../lib/logger';

beforeEach(() => {
    vi.clearAllMocks();
    api.sendLog.mockResolvedValue({ status: 200 });
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

    it('logger.info sends level "info" through the API client', async () => {
        await logger.info('hello world');
        expect(api.sendLog).toHaveBeenCalledWith(
            expect.objectContaining({ level: 'info', message: 'hello world', source: 'Frontend' })
        );
    });

    it('logger.warn sends level "warn"', async () => {
        await logger.warn('something suspicious');
        expect(api.sendLog).toHaveBeenCalledWith(
            expect.objectContaining({ level: 'warn', message: 'something suspicious' })
        );
    });

    it('logger.error sends level "error" and optional stack', async () => {
        const stack = 'Error at line 42';
        await logger.error('crash', stack);
        expect(api.sendLog).toHaveBeenCalledWith(
            expect.objectContaining({ level: 'error', message: 'crash', stack })
        );
    });

    it('logger.error works without a stack argument', async () => {
        await logger.error('no stack error');
        expect(api.sendLog).toHaveBeenCalledWith(
            expect.objectContaining({ level: 'error', message: 'no stack error', stack: null })
        );
    });

    it('serializes non-string messages to JSON', async () => {
        await logger.log('info', { key: 'value' });
        expect(api.sendLog).toHaveBeenCalledWith(
            expect.objectContaining({ message: '{"key":"value"}' })
        );
    });

    it('does not throw when the backend post fails', async () => {
        api.sendLog.mockRejectedValueOnce(new Error('Network down'));
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
