import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

vi.mock('../lib/api', () => ({
    default: {
        getConfig: vi.fn().mockResolvedValue({ data: { provider: 'openai' } }),
        listLocalModels: vi.fn().mockResolvedValue({ data: [] }),
        saveConfig: vi.fn(),
        search: vi.fn(),
        streamAnswer: vi.fn(),
    },
}));

import SearchView from '../components/SearchView';
import api from '../lib/api';

describe('SearchView motion states', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        api.getConfig.mockResolvedValue({ data: { provider: 'openai' } });
        api.listLocalModels.mockResolvedValue({ data: [] });
    });

    it('shows the empty-state visual before the first search', async () => {
        render(<SearchView />);

        expect(screen.getByText('Search your documents.')).toBeDefined();
        expect(screen.getByText(/Ask natural language questions across your indexed files/i)).toBeDefined();
    });

    it('shows shimmer skeleton loading state while a search is in flight', async () => {
        api.search.mockReturnValue(new Promise(() => {}));
        const { container } = render(<SearchView />);

        fireEvent.change(screen.getByLabelText('Search'), { target: { value: 'transformer architecture' } });
        fireEvent.submit(screen.getByLabelText('Search').closest('form'));

        await waitFor(() => {
            const shimmers = container.querySelectorAll('.shimmer');
            expect(shimmers.length).toBeGreaterThan(0);
        });
    });

    it('toggles model selection dropdown when clicking model selector button', async () => {
        render(<SearchView />);

        const modelBtn = await screen.findByRole('button', { name: /openai/i });
        fireEvent.click(modelBtn);

        expect(screen.getByText('Cloud')).toBeDefined();
        expect(screen.getByText('External')).toBeDefined();
    });
});
