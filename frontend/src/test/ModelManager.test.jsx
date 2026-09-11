import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import { ToastProvider } from '../components/Toast';
import ModelManager from '../components/ModelManager';
import api from '../lib/api';

vi.mock('../lib/api', () => ({
    default: {
        listAvailableModels: vi.fn(),
        listLocalModels: vi.fn(),
        modelDownloadStatus: vi.fn(),
        downloadModel: vi.fn(),
        deleteModel: vi.fn(),
    }
}));

const mockAvailable = [
    {
        id: 'gemma-2-2b-it.Q4_K_M',
        name: 'Gemma 2 2B Instruct (Google)',
        description: 'Google ultra-fast 2.6B model.',
        size: '1.7 GB',
        ram_required: 4,
        category: '8gb',
        ram_tier: '8GB RAM',
        speed: 'Ultra-Fast',
        intelligence: 'Very High',
        quantization: 'Q4_K_M',
        use_case: 'Fast QA, Document Search',
        recommended: true,
        url: 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf'
    },
    {
        id: 'Qwen2.5-7B-Instruct.Q4_K_M',
        name: 'Qwen 2.5 7B Instruct (Alibaba)',
        description: 'Worldwide leader among open 7B models.',
        size: '4.7 GB',
        ram_required: 9,
        category: '16gb',
        ram_tier: '16GB RAM',
        speed: 'Fast',
        intelligence: 'State-of-the-Art',
        quantization: 'Q4_K_M',
        use_case: 'Deep Document Analysis',
        recommended: true,
        url: 'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf'
    },
    {
        id: 'Qwen2.5-14B-Instruct.Q4_K_M',
        name: 'Qwen 2.5 14B Instruct (Alibaba)',
        description: 'Frontier-level intelligence.',
        size: '9.0 GB',
        ram_required: 16,
        category: '32gb',
        ram_tier: '32GB+ RAM',
        speed: 'Balanced',
        intelligence: 'Frontier-Grade',
        quantization: 'Q4_K_M',
        use_case: 'Complex Research',
        recommended: false,
        url: 'https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf'
    }
];

const mockLocal = [
    {
        id: 'gemma-2-2b-it.Q4_K_M',
        name: 'Gemma 2 2B Instruct (Google)',
        filename: 'gemma-2-2b-it.Q4_K_M.gguf',
        path: '/models/gemma-2-2b-it.Q4_K_M.gguf',
        size: 1708582752,
        ram_tier: '8GB RAM',
        ram_required: 4,
        speed: 'Ultra-Fast'
    }
];

describe('ModelManager Component', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        api.listAvailableModels.mockResolvedValue({ data: mockAvailable });
        api.listLocalModels.mockResolvedValue({ data: mockLocal });
        api.modelDownloadStatus.mockResolvedValue({ data: { downloading: false } });
    });

    it('renders RAM tier filter buttons instead of small/medium/large', async () => {
        render(
            <ToastProvider>
                <ModelManager activeModelPath="/models/gemma-2-2b-it.Q4_K_M.gguf" onSelectModel={vi.fn()} />
            </ToastProvider>
        );

        await waitFor(() => {
            expect(screen.getByText('Compatible with 8GB')).toBeDefined();
            expect(screen.getByText('Compatible with 16GB')).toBeDefined();
            expect(screen.getByText('Compatible with 32GB+')).toBeDefined();
            expect(screen.getByText('All Models')).toBeDefined();
        });

        // Ensure legacy category names are not in the filter buttons
        expect(screen.queryByRole('button', { name: /^small$/i })).toBeNull();
        expect(screen.queryByRole('button', { name: /^medium$/i })).toBeNull();
        expect(screen.queryByRole('button', { name: /^large$/i })).toBeNull();
    });

    it('filters models by RAM tier when clicking a filter button', async () => {
        render(
            <ToastProvider>
                <ModelManager activeModelPath="" onSelectModel={vi.fn()} />
            </ToastProvider>
        );

        await waitFor(() => {
            expect(screen.getAllByText('Gemma 2 2B Instruct (Google)').length).toBeGreaterThanOrEqual(1);
            expect(screen.getByText('Qwen 2.5 7B Instruct (Alibaba)')).toBeDefined();
            expect(screen.getByText('Qwen 2.5 14B Instruct (Alibaba)')).toBeDefined();
        });

        // Click Compatible with 8GB
        fireEvent.click(screen.getByText('Compatible with 8GB'));
        expect(screen.getAllByText('Gemma 2 2B Instruct (Google)').length).toBeGreaterThanOrEqual(1);
        expect(screen.queryByText('Qwen 2.5 7B Instruct (Alibaba)')).toBeNull();
        expect(screen.queryByText('Qwen 2.5 14B Instruct (Alibaba)')).toBeNull();

        // Click Compatible with 16GB
        fireEvent.click(screen.getByText('Compatible with 16GB'));
        expect(screen.queryByText('Qwen 2.5 14B Instruct (Alibaba)')).toBeNull();
        expect(screen.getByText('Qwen 2.5 7B Instruct (Alibaba)')).toBeDefined();

        // Click Compatible with 32GB+
        fireEvent.click(screen.getByText('Compatible with 32GB+'));
        expect(screen.queryByText('Qwen 2.5 7B Instruct (Alibaba)')).toBeNull();
        expect(screen.getByText('Qwen 2.5 14B Instruct (Alibaba)')).toBeDefined();
    });

    it('displays RAM tier badges and speed/intelligence chips', async () => {
        render(
            <ToastProvider>
                <ModelManager activeModelPath="" onSelectModel={vi.fn()} />
            </ToastProvider>
        );

        await waitFor(() => {
            expect(screen.getAllByText('8GB RAM').length).toBeGreaterThanOrEqual(1);
            expect(screen.getByText('16GB RAM')).toBeDefined();
            expect(screen.getByText('32GB+ RAM')).toBeDefined();
            expect(screen.getAllByText(/Ultra-Fast/).length).toBeGreaterThanOrEqual(1);
            expect(screen.getByText(/State-of-the-Art/)).toBeDefined();
            expect(screen.getByText(/Frontier-Grade/)).toBeDefined();
        });
    });

    it('triggers model download when download button is clicked', async () => {
        api.downloadModel.mockResolvedValue({ data: { success: true } });

        render(
            <ToastProvider>
                <ModelManager activeModelPath="" onSelectModel={vi.fn()} />
            </ToastProvider>
        );

        await waitFor(() => {
            expect(screen.getByText('Qwen 2.5 7B Instruct (Alibaba)')).toBeDefined();
        });

        const downloadButtons = screen.getAllByRole('button', { name: /download/i });
        fireEvent.click(downloadButtons[0]);

        expect(api.downloadModel).toHaveBeenCalled();
    });
});
