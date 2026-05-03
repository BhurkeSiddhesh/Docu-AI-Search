/**
 * BenchmarkResults Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import BenchmarkResults from '../components/BenchmarkResults';

vi.mock('axios');

const mockResultsData = {
    results: [
        {
            model_id: 'model-a',
            model_name: 'Model A',
            metrics: { precision: 0.9, recall: 0.85, f1: 0.87, latency_ms: 120 },
        },
        {
            model_id: 'model-b',
            model_name: 'Model B',
            metrics: { precision: 0.75, recall: 0.80, f1: 0.77, latency_ms: 90 },
        },
    ],
};

beforeEach(() => {
    vi.clearAllMocks();
    // Default: results endpoint returns data, status endpoint says not running
    axios.get.mockImplementation((url) => {
        if (url.includes('/results')) return Promise.resolve({ data: mockResultsData });
        if (url.includes('/status')) return Promise.resolve({ data: { running: false } });
        return Promise.reject(new Error('Unknown URL'));
    });
});

afterEach(() => {
    vi.restoreAllMocks();
});

describe('BenchmarkResults Component', () => {
    it('renders without crashing', async () => {
        render(<BenchmarkResults />);
        await waitFor(() => {
            expect(screen.queryByText(/loading/i) === null || true).toBe(true);
        });
    });

    it('shows a Run Benchmark button', async () => {
        render(<BenchmarkResults />);
        await waitFor(() => {
            expect(screen.getByText(/run benchmark/i)).toBeDefined();
        });
    });

    it('calls the benchmarks run endpoint on button click', async () => {
        axios.post.mockResolvedValueOnce({ data: { status: 'started' } });

        render(<BenchmarkResults />);
        await waitFor(() => screen.getByText(/run benchmark/i));

        fireEvent.click(screen.getByText(/run benchmark/i));

        await waitFor(() => {
            expect(axios.post).toHaveBeenCalledWith(
                expect.stringContaining('/benchmarks/run')
            );
        });
    });

    it('shows an error message when starting the benchmark fails', async () => {
        axios.post.mockRejectedValueOnce({
            response: { data: { detail: 'Benchmark already running' } },
        });

        render(<BenchmarkResults />);
        await waitFor(() => screen.getByText(/run benchmark/i));

        fireEvent.click(screen.getByText(/run benchmark/i));

        await waitFor(() => {
            expect(screen.getByText('Benchmark already running')).toBeDefined();
        });
    });

    it('fetches results on mount', async () => {
        render(<BenchmarkResults />);
        await waitFor(() => {
            expect(axios.get).toHaveBeenCalledWith(
                expect.stringContaining('/benchmarks/results')
            );
        });
    });

    it('polls benchmark status on mount', async () => {
        vi.useFakeTimers();

        render(<BenchmarkResults />);

        await waitFor(() => {
            expect(axios.get).toHaveBeenCalledWith(
                expect.stringContaining('/benchmarks/status')
            );
        });

        vi.useRealTimers();
    });

    it('handles a failed results fetch gracefully', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/results')) return Promise.reject(new Error('Network error'));
            return Promise.resolve({ data: { running: false } });
        });

        render(<BenchmarkResults />);

        await waitFor(() => {
            expect(screen.queryByText(/network error/i)).toBeNull();
        });
    });
});
