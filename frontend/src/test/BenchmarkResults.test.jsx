/**
 * BenchmarkResults Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
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
        await act(async () => {
            render(<BenchmarkResults />);
        });
        expect(document.body).toBeDefined();
    });

    it('shows Start Benchmark button after loading', async () => {
        await act(async () => {
            render(<BenchmarkResults />);
        });
        expect(screen.getByText('Start Benchmark')).toBeDefined();
    });

    it('calls the benchmarks run endpoint on button click', async () => {
        axios.post.mockResolvedValueOnce({ data: { status: 'started' } });

        await act(async () => {
            render(<BenchmarkResults />);
        });

        await act(async () => {
            fireEvent.click(screen.getByText('Start Benchmark'));
        });

        expect(axios.post).toHaveBeenCalledWith(
            expect.stringContaining('/benchmarks/run')
        );
    });

    it('shows an error message when starting the benchmark fails', async () => {
        axios.post.mockRejectedValueOnce({
            response: { data: { detail: 'Benchmark already running' } },
        });

        await act(async () => {
            render(<BenchmarkResults />);
        });

        await act(async () => {
            fireEvent.click(screen.getByText('Start Benchmark'));
        });

        expect(screen.getByText('Benchmark already running')).toBeDefined();
    });

    it('fetches results on mount', async () => {
        await act(async () => {
            render(<BenchmarkResults />);
        });

        expect(axios.get).toHaveBeenCalledWith(
            expect.stringContaining('/benchmarks/results')
        );
    });

    it('polls benchmark status after interval fires', async () => {
        vi.useFakeTimers();

        await act(async () => {
            render(<BenchmarkResults />);
        });

        // Advance timers to trigger setInterval(checkStatus, 2000)
        await act(async () => {
            vi.advanceTimersByTime(2001);
        });

        vi.useRealTimers();

        expect(axios.get).toHaveBeenCalledWith(
            expect.stringContaining('/benchmarks/status')
        );
    });

    it('handles a failed results fetch gracefully without crashing', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/results')) return Promise.reject(new Error('Network error'));
            return Promise.resolve({ data: { running: false } });
        });

        await act(async () => {
            render(<BenchmarkResults />);
        });

        expect(screen.queryByText(/exception/i)).toBeNull();
    });
});
