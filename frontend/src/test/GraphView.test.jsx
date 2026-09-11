import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import React from 'react';
import axios from 'axios';
import { FileText, FileSpreadsheet, Presentation, FileType } from 'lucide-react';

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

import GraphView, { getDocIcon } from '../components/GraphView';
import api from '../lib/api';

describe('GraphView file icons and layout', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('resolves correct icon component for different file types', () => {
        expect(getDocIcon('.pdf')).toBe(FileText);
        expect(getDocIcon('pdf')).toBe(FileText);
        expect(getDocIcon('.docx')).toBe(FileText);
        expect(getDocIcon('.txt')).toBe(FileText);
        expect(getDocIcon('.md')).toBe(FileText);

        expect(getDocIcon('.xlsx')).toBe(FileSpreadsheet);
        expect(getDocIcon('.xls')).toBe(FileSpreadsheet);
        expect(getDocIcon('.csv')).toBe(FileSpreadsheet);

        expect(getDocIcon('.pptx')).toBe(Presentation);
        expect(getDocIcon('.ppt')).toBe(Presentation);

        expect(getDocIcon('.unknown')).toBe(FileType);
        expect(getDocIcon('')).toBe(FileType);
    });

    it('renders small file icons (<= 12px) for document nodes in graph view', async () => {
        const mockGraph = {
            nodes: [
                {
                    id: '/path/to/finance.xlsx',
                    type: 'document',
                    label: 'finance.xlsx',
                    metadata: JSON.stringify({ file_type: '.xlsx', chunks: 12 }),
                },
                {
                    id: '/path/to/slides.pptx',
                    type: 'document',
                    label: 'slides.pptx',
                    metadata: JSON.stringify({ file_type: '.pptx', chunks: 5 }),
                },
                {
                    id: '/path/to/report.pdf',
                    type: 'document',
                    label: 'report.pdf',
                    metadata: JSON.stringify({ file_type: '.pdf', chunks: 1 }),
                },
                {
                    id: 'kw_revenue',
                    type: 'keyword',
                    label: 'revenue',
                    metadata: '{}',
                },
            ],
            edges: [
                { source_id: '/path/to/finance.xlsx', target_id: 'kw_revenue', relation_type: 'mentions', weight: 3 },
            ],
        };

        vi.spyOn(api, 'getGraph').mockResolvedValueOnce({ data: mockGraph });

        render(<GraphView />);

        await waitFor(() => {
            expect(screen.getByText('Knowledge graph')).toBeDefined();
        });

        // Verify document labels are displayed
        expect(screen.getByText('finance.xlsx')).toBeDefined();
        expect(screen.getByText('slides.pptx')).toBeDefined();
        expect(screen.getByText('report.pdf')).toBeDefined();
        expect(screen.getByText('revenue')).toBeDefined();

        // Verify file icons rendered inside the document nodes
        const icons = screen.getAllByTestId('graph-file-icon');
        expect(icons).toHaveLength(3);

        // Verify each file icon is small (width and height <= 12px)
        icons.forEach((icon) => {
            const width = parseFloat(icon.getAttribute('width'));
            const height = parseFloat(icon.getAttribute('height'));
            expect(width).toBeLessThanOrEqual(12);
            expect(height).toBeLessThanOrEqual(12);
            expect(width).toBeGreaterThanOrEqual(8);
            expect(height).toBeGreaterThanOrEqual(8);
        });
    });

    it('renders empty graph art when no nodes exist', async () => {
        vi.spyOn(api, 'getGraph').mockResolvedValueOnce({ data: { nodes: [], edges: [] } });

        render(<GraphView />);

        await waitFor(() => {
            expect(screen.getByText('No knowledge graph yet')).toBeDefined();
        });
    });

    it('renders error state when graph loading fails', async () => {
        vi.spyOn(api, 'getGraph').mockRejectedValueOnce(new Error('Network error'));

        render(<GraphView />);

        await waitFor(() => {
            expect(screen.getByText('Graph unavailable')).toBeDefined();
        });
    });
});
