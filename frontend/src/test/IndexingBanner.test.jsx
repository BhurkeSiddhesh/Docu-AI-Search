import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import React from 'react';

vi.mock('../lib/api', () => ({
    default: {
        getIndexStatus: vi.fn(),
    },
}));

import IndexingBanner from '../components/IndexingBanner';
import api from '../lib/api';

class MockWebSocket {
    static instances = [];
    constructor() { MockWebSocket.instances.push(this); }
    close() {}
}

describe('IndexingBanner', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        MockWebSocket.instances = [];
        global.WebSocket = MockWebSocket;
    });

    afterEach(() => {
        delete global.WebSocket;
    });

    it('renders nothing when not running and no error', async () => {
        api.getIndexStatus.mockResolvedValue({ data: { running: false, error: null } });
        const { container } = render(<IndexingBanner />);
        await waitFor(() => {}, { timeout: 100 });
        expect(container.firstChild).toBeNull();
    });

    it('shows progress, stage chip, and file name when running', async () => {
        api.getIndexStatus.mockResolvedValue({
            data: {
                running: true,
                progress: 42,
                current_file: 'report.pdf',
                stage: 'embedding',
                total_files: 20,
                processed_files: 8,
                eta_seconds: 90,
                error: null,
            },
        });
        render(<IndexingBanner />);

        await waitFor(() => {
            expect(screen.getByText('42%')).toBeDefined();
        });

        // Stage chip
        expect(screen.getByText('Embedding')).toBeDefined();
        // File counter
        expect(screen.getByText(/8 \/ 20 files/)).toBeDefined();
        // ETA
        expect(screen.getByText(/remaining/)).toBeDefined();
    });

    it('shows shimmer bar (no filled bar) when progress is 0%', async () => {
        api.getIndexStatus.mockResolvedValue({
            data: {
                running: true,
                progress: 0,
                current_file: 'Scanning files…',
                stage: 'starting',
                total_files: 0,
                processed_files: 0,
                eta_seconds: null,
                error: null,
            },
        });
        render(<IndexingBanner />);

        await waitFor(() => {
            expect(screen.getByText('0%')).toBeDefined();
        });

        expect(screen.getByText('Starting…')).toBeDefined();
        // No filled progress div (no style width attribute present in shimmer mode)
        const filledBar = document.querySelector('[style*="width"]');
        expect(filledBar).toBeNull();
    });

    it('expands to show full file path on chevron click', async () => {
        api.getIndexStatus.mockResolvedValue({
            data: {
                running: true,
                progress: 30,
                current_file: '/very/long/path/to/document.pdf',
                stage: 'extracting',
                total_files: 5,
                processed_files: 2,
                eta_seconds: null,
                error: null,
            },
        });
        render(<IndexingBanner />);

        await waitFor(() => {
            expect(screen.getByText('30%')).toBeDefined();
        });

        // The filename should be truncated by default (in a truncate div)
        const expandBtn = screen.getByLabelText('Expand detail');
        fireEvent.click(expandBtn);

        // After expand the file should be shown without truncation
        expect(screen.getByLabelText('Collapse detail')).toBeDefined();
    });

    it('shows error state when error is present', async () => {
        api.getIndexStatus.mockResolvedValue({
            data: { running: false, error: 'No files found', progress: 0 },
        });
        render(<IndexingBanner />);

        await waitFor(() => {
            expect(screen.getByText('Indexing failed')).toBeDefined();
        });
        expect(screen.getByText('No files found')).toBeDefined();
    });
});
