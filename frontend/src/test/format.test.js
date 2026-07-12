/**
 * Tests for frontend/src/lib/format.js
 *
 * Covers formatBytes, formatRelative, and fileExt utility functions.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { formatBytes, formatRelative, fileExt } from '../lib/format';

// ── formatBytes ───────────────────────────────────────────────────────────────

describe('formatBytes', () => {
    it('returns "0 B" for zero bytes', () => {
        expect(formatBytes(0)).toBe('0 B');
    });

    it('returns "0 B" for null/undefined', () => {
        expect(formatBytes(null)).toBe('0 B');
        expect(formatBytes(undefined)).toBe('0 B');
    });

    it('returns "0 B" for negative values', () => {
        expect(formatBytes(-100)).toBe('0 B');
    });

    it('formats bytes below 1 KB correctly', () => {
        expect(formatBytes(512)).toBe('512 B');
    });

    it('formats 1 KB correctly', () => {
        expect(formatBytes(1024)).toBe('1 KB');
    });

    it('formats values in the KB range', () => {
        expect(formatBytes(2048)).toBe('2 KB');
        expect(formatBytes(1536)).toBe('1.5 KB');
    });

    it('formats 1 MB correctly', () => {
        expect(formatBytes(1024 * 1024)).toBe('1 MB');
    });

    it('formats values in the MB range', () => {
        expect(formatBytes(1024 * 1024 * 2.5)).toBe('2.5 MB');
    });

    it('formats 1 GB correctly', () => {
        expect(formatBytes(1024 ** 3)).toBe('1 GB');
    });

    it('formats TB correctly', () => {
        expect(formatBytes(1024 ** 4)).toBe('1 TB');
    });
});

// ── formatRelative ────────────────────────────────────────────────────────────

describe('formatRelative', () => {
    let now;

    beforeEach(() => {
        now = Date.now();
        vi.useFakeTimers();
        vi.setSystemTime(now);
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it('returns empty string for falsy timestamp', () => {
        expect(formatRelative(null)).toBe('');
        expect(formatRelative(undefined)).toBe('');
        expect(formatRelative(0)).toBe('');
    });

    it('returns "just now" for timestamps within the last minute', () => {
        const thirtySecondsAgo = new Date(now - 30_000).toISOString();
        expect(formatRelative(thirtySecondsAgo)).toBe('just now');
    });

    it('returns minutes ago for timestamps 1–59 minutes old', () => {
        const fiveMinutesAgo = new Date(now - 5 * 60_000).toISOString();
        expect(formatRelative(fiveMinutesAgo)).toBe('5m ago');
    });

    it('returns "1m ago" for exactly 1 minute ago', () => {
        const oneMinuteAgo = new Date(now - 60_000).toISOString();
        expect(formatRelative(oneMinuteAgo)).toBe('1m ago');
    });

    it('returns hours ago for timestamps 1–23 hours old', () => {
        const twoHoursAgo = new Date(now - 2 * 3_600_000).toISOString();
        expect(formatRelative(twoHoursAgo)).toBe('2h ago');
    });

    it('returns "1h ago" for exactly 1 hour ago', () => {
        const oneHourAgo = new Date(now - 3_600_000).toISOString();
        expect(formatRelative(oneHourAgo)).toBe('1h ago');
    });

    it('returns days ago for timestamps 1–6 days old', () => {
        const threeDaysAgo = new Date(now - 3 * 86_400_000).toISOString();
        expect(formatRelative(threeDaysAgo)).toBe('3d ago');
    });

    it('returns "1d ago" for exactly 1 day ago', () => {
        const oneDayAgo = new Date(now - 86_400_000).toISOString();
        expect(formatRelative(oneDayAgo)).toBe('1d ago');
    });

    it('returns a formatted date string for timestamps older than 7 days', () => {
        const twoWeeksAgo = new Date(now - 14 * 86_400_000).toISOString();
        const result = formatRelative(twoWeeksAgo);
        // Should be a locale-formatted date, not a relative string
        expect(result).not.toContain('ago');
        expect(result).not.toBe('just now');
        expect(result.length).toBeGreaterThan(0);
    });
});

// ── fileExt ────────────────────────────────────────────────────────────────────

describe('fileExt', () => {
    it('returns empty string for null/undefined', () => {
        expect(fileExt(null)).toBe('');
        expect(fileExt(undefined)).toBe('');
    });

    it('returns empty string for empty string', () => {
        expect(fileExt('')).toBe('');
    });

    it('extracts and uppercases the extension', () => {
        expect(fileExt('document.pdf')).toBe('PDF');
        expect(fileExt('report.docx')).toBe('DOCX');
        expect(fileExt('data.xlsx')).toBe('XLSX');
        expect(fileExt('slides.pptx')).toBe('PPTX');
        expect(fileExt('notes.txt')).toBe('TXT');
    });

    it('handles filenames with multiple dots', () => {
        expect(fileExt('my.report.final.pdf')).toBe('PDF');
    });

    it('handles filenames with no extension', () => {
        expect(fileExt('README')).toBe('');
    });

    it('handles dotfiles (hidden files starting with a dot)', () => {
        // .gitignore has extension "gitignore"
        expect(fileExt('.gitignore')).toBe('GITIGNORE');
    });

    it('handles mixed-case extensions', () => {
        expect(fileExt('file.PDF')).toBe('PDF');
        expect(fileExt('image.Jpg')).toBe('JPG');
    });
});
