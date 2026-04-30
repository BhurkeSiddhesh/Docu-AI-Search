/**
 * SearchHistory Component Tests
 * 
 * Tests for the SearchHistory component including list rendering
 * and delete interactions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import SearchHistory from '../components/SearchHistory'
import axios from 'axios'

// Mock axios
vi.mock('axios')

describe('SearchHistory Component', () => {
    const mockHistory = [
        { id: 1, query: 'react hooks', timestamp: new Date().toISOString(), result_count: 5 },
        { id: 2, query: 'vitest testing', timestamp: new Date().toISOString(), result_count: 12 }
    ]

    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('renders nothing when not open', () => {
        render(<SearchHistory isOpen={false} />)
        expect(screen.queryByText('History')).toBeNull()
    })

    it('fetches and renders history when opened', async () => {
        axios.get.mockResolvedValueOnce({ data: mockHistory })

        render(<SearchHistory isOpen={true} onClose={vi.fn()} onSelectQuery={vi.fn()} />)

        expect(axios.get).toHaveBeenCalledWith('http://localhost:8000/api/search/history')

        // Wait for loading to finish and items to appear
        await waitFor(() => {
            expect(screen.getByText('react hooks')).toBeDefined()
            expect(screen.getByText('vitest testing')).toBeDefined()
        })
    })

    it('calls onSelectQuery and onClose when an item is clicked', async () => {
        axios.get.mockResolvedValueOnce({ data: mockHistory })
        const onSelectQuery = vi.fn()
        const onClose = vi.fn()

        render(<SearchHistory isOpen={true} onClose={onClose} onSelectQuery={onSelectQuery} />)

        await waitFor(() => screen.getByText('react hooks'))

        fireEvent.click(screen.getByText('react hooks'))

        expect(onSelectQuery).toHaveBeenCalledWith('react hooks')
        expect(onClose).toHaveBeenCalled()
    })

    it('deletes an item when delete button is clicked', async () => {
        axios.get.mockResolvedValueOnce({ data: mockHistory })
        axios.delete.mockResolvedValueOnce({ data: { success: true } })

        render(<SearchHistory isOpen={true} onClose={vi.fn()} onSelectQuery={vi.fn()} />)

        await waitFor(() => screen.getByText('react hooks'))

        // Use the new aria-label for robustness
        const deleteButtons = screen.getAllByLabelText('Delete history item')
        fireEvent.click(deleteButtons[0])

        expect(axios.delete).toHaveBeenCalledWith('http://localhost:8000/api/search/history/1')

        // Verify item is removed from view (optimistic update or after re-render)
        await waitFor(() => {
            expect(screen.queryByText('react hooks')).toBeNull()
        })
    })

    it('handles empty history', async () => {
        axios.get.mockResolvedValueOnce({ data: [] })

        render(<SearchHistory isOpen={true} />)

        await waitFor(() => {
            expect(screen.getByText(/No historical data available/i)).toBeDefined()
        })
    })
})
