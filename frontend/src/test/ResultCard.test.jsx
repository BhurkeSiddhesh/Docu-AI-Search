/**
 * Tests for frontend/src/components/ResultCard.jsx
 *
 * Verifies rendering of file metadata, tags, preview toggle behaviour,
 * and error handling when the API returns an error.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import axios from 'axios';

vi.mock('axios', () => {
    const mockClient = {
        get: vi.fn(),
        post: vi.fn(),
        delete: vi.fn(),
        interceptors: { request: { use: vi.fn() }, response: { use: vi.fn() } },
    };
    return {
        default: { ...mockClient, create: vi.fn(() => mockClient) },
    };
});

import ResultCard from '../components/ResultCard';
import { ToastProvider } from '../components/Toast';

const mockClient = axios.create();

function renderCard(result) {
    return render(
        <ToastProvider>
            <ResultCard result={result} />
        </ToastProvider>
    );
}

const baseResult = {
    file_name: 'annual_report.pdf',
    file_path: '/docs/annual_report.pdf',
    document: 'This is the extracted text content of the document.',
    distance: 0.12,
    tags: ['finance', 'revenue', 'annual'],
    faiss_idx: 0,
};

beforeEach(() => {
    vi.clearAllMocks();
    mockClient.get.mockResolvedValue({ data: { preview: 'Preview text here.' } });
    mockClient.post.mockResolvedValue({ data: {} });
});

// ── Basic rendering ───────────────────────────────────────────────────────────

describe('ResultCard rendering', () => {
    it('displays the file name', () => {
        renderCard(baseResult);
        expect(screen.getByText('annual_report.pdf')).toBeDefined();
    });

    it('displays the file path', () => {
        renderCard(baseResult);
        expect(screen.getByText('/docs/annual_report.pdf')).toBeDefined();
    });

    it('displays the document text snippet', () => {
        renderCard(baseResult);
        expect(screen.getByText(/extracted text content/i)).toBeDefined();
    });

    it('shows the file extension chip', () => {
        renderCard(baseResult);
        const chips = screen.getAllByText('pdf');
        expect(chips.length).toBeGreaterThan(0);
    });

    it('renders "Unknown document" when file_name is missing', () => {
        renderCard({ ...baseResult, file_name: undefined });
        expect(screen.getByText('Unknown document')).toBeDefined();
    });

    it('renders tags up to the first 6', () => {
        const result = {
            ...baseResult,
            tags: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        };
        renderCard(result);
        // Up to 6 tags visible
        expect(screen.getByText('a')).toBeDefined();
        expect(screen.getByText('f')).toBeDefined();
        expect(screen.queryByText('g')).toBeNull();
    });

    it('renders no tag chips when tags array is empty', () => {
        renderCard({ ...baseResult, tags: [] });
        expect(screen.queryByText('finance')).toBeNull();
    });

    it('displays the summary when provided', () => {
        const result = { ...baseResult, summary: 'A concise summary of the document.' };
        renderCard(result);
        expect(screen.getByText('A concise summary of the document.')).toBeDefined();
    });

    it('hides the file path row when file_path is missing', () => {
        renderCard({ ...baseResult, file_path: undefined });
        expect(screen.queryByText('/docs/annual_report.pdf')).toBeNull();
    });
});

// ── Preview toggle ────────────────────────────────────────────────────────────

describe('ResultCard preview', () => {
    it('does not show preview panel initially', () => {
        renderCard(baseResult);
        expect(screen.queryByText('Preview')).toBeNull();
    });

    it('shows preview panel when preview button is clicked', async () => {
        renderCard(baseResult);
        const previewBtn = screen.getByTitle('Preview');
        await act(async () => {
            fireEvent.click(previewBtn);
        });
        await waitFor(() => {
            expect(screen.getByText('Preview')).toBeDefined();
        });
    });

    it('calls api.previewFile when preview is toggled open', async () => {
        renderCard(baseResult);
        const previewBtn = screen.getByTitle('Preview');
        await act(async () => {
            fireEvent.click(previewBtn);
        });
        await waitFor(() => {
            expect(mockClient.get).toHaveBeenCalledWith(
                expect.stringContaining('files/preview')
            );
        });
    });

    it('displays fetched preview text', async () => {
        mockClient.get.mockResolvedValue({ data: { preview: 'Fetched preview content.' } });
        renderCard(baseResult);
        await act(async () => {
            fireEvent.click(screen.getByTitle('Preview'));
        });
        await waitFor(() => {
            expect(screen.getByText('Fetched preview content.')).toBeDefined();
        });
    });

    it('shows fallback message when preview API returns no text', async () => {
        mockClient.get.mockResolvedValue({ data: { preview: '' } });
        renderCard(baseResult);
        await act(async () => {
            fireEvent.click(screen.getByTitle('Preview'));
        });
        await waitFor(() => {
            expect(screen.getByText('No preview available.')).toBeDefined();
        });
    });

    it('shows error fallback when preview API fails', async () => {
        mockClient.get.mockRejectedValue(new Error('Network error'));
        renderCard(baseResult);
        await act(async () => {
            fireEvent.click(screen.getByTitle('Preview'));
        });
        await waitFor(() => {
            expect(screen.getByText('Could not load preview.')).toBeDefined();
        });
    });

    it('closes preview when the close button is clicked', async () => {
        renderCard(baseResult);
        await act(async () => {
            fireEvent.click(screen.getByTitle('Preview'));
        });
        await waitFor(() => screen.getByText('Preview'));
        // The close button is a sibling of the "Preview" label in the flex header
        const previewLabel = screen.getByText('Preview');
        const headerRow = previewLabel.parentElement;
        const closeBtn = headerRow.querySelector('button');
        await act(async () => {
            fireEvent.click(closeBtn);
        });
        expect(screen.queryByText('Preview')).toBeNull();
    });

    it('does not re-fetch preview when toggling close then open again', async () => {
        renderCard(baseResult);
        const previewBtn = screen.getByTitle('Preview');

        // Open for the first time
        await act(async () => { fireEvent.click(previewBtn); });
        await waitFor(() => screen.getByText('Preview'));

        // Close via the eye button toggle
        await act(async () => { fireEvent.click(previewBtn); });
        expect(screen.queryByText('Preview')).toBeNull();

        // Re-open — should NOT trigger another API call
        await act(async () => { fireEvent.click(previewBtn); });
        await waitFor(() => screen.getByText('Preview'));

        // API should have been called exactly once
        const previewCalls = mockClient.get.mock.calls.filter(
            (args) => args[0] && args[0].includes('preview')
        );
        expect(previewCalls.length).toBe(1);
    });
});

// ── Open file action ──────────────────────────────────────────────────────────

describe('ResultCard open file', () => {
    it('calls api.openFile when the file name button is clicked', async () => {
        renderCard(baseResult);
        await act(async () => {
            fireEvent.click(screen.getByText('annual_report.pdf'));
        });
        expect(mockClient.post).toHaveBeenCalledWith(
            '/open-file',
            { path: '/docs/annual_report.pdf' }
        );
    });

    it('calls api.openFile when the external link button is clicked', async () => {
        renderCard(baseResult);
        await act(async () => {
            fireEvent.click(screen.getByTitle('Open in system viewer'));
        });
        expect(mockClient.post).toHaveBeenCalledWith(
            '/open-file',
            { path: '/docs/annual_report.pdf' }
        );
    });
});
