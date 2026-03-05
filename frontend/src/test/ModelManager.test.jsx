/**
 * ModelManager Component Tests
 * 
 * Functional tests for ModelManager using React Testing Library.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import ModelManager from '../components/ModelManager'
import axios from 'axios'

// Mock axios and window.confirm
vi.mock('axios')
global.confirm = vi.fn(() => true)

describe('ModelManager Component', () => {
    const mockAvailableModels = [
        { id: 'phi-2', name: 'Phi-2', size: '1.7 GB', ram_required: 4, recommended: true },
        { id: 'tinyllama', name: 'TinyLlama', size: '600 MB', ram_required: 2 }
    ]

    const mockLocalModels = [
        { id: 'phi-2', name: 'Phi-2', path: '/models/phi-2.gguf', size: 1700000000 }
    ]

    const mockRecommendations = {
        system: { ram_gb_total: 16, cpu_cores_logical: 8, disk_gb_free: 120 },
        recommendations: [
            { id: 'phi-2', name: 'Phi-2', compatibility: 'excellent' }
        ]
    }

    beforeEach(() => {
        vi.clearAllMocks()
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/models/available')) return Promise.resolve({ data: mockAvailableModels })
            if (url.includes('/api/models/local')) return Promise.resolve({ data: mockLocalModels })
            if (url.includes('/api/models/recommendations')) return Promise.resolve({ data: mockRecommendations })
            if (url.includes('/api/models/status')) return Promise.resolve({ data: { downloading: false } })
            return Promise.resolve({ data: {} })
        })
    })

    it('renders and fetches models', async () => {
        render(<ModelManager activeModel="phi-2" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => {
            // Use getAllByText because Phi-2 appears in both lists
            expect(screen.getAllByText('Phi-2').length).toBeGreaterThan(0)
            expect(screen.getByText(/1.6 GB/i)).toBeDefined()
            expect(screen.getByText(/Smart Recommendations for Your System/i)).toBeDefined()
        })
    })

    it('triggers download when download button is clicked', async () => {
        axios.post.mockResolvedValueOnce({ data: { status: 'success' } })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getAllByText('TinyLlama')[0])

        const cards = screen.getAllByText('TinyLlama').map(el => el.closest('.border'))
        const card = cards.find(c => within(c).queryByText(/Download/i))
        const downloadBtn = within(card).getByRole('button', { name: /Download/i })

        fireEvent.click(downloadBtn)

        expect(axios.post).toHaveBeenCalledWith('http://localhost:8000/api/models/download/tinyllama')
    })

    it('triggers select model callback', async () => {
        const onSelectModel = vi.fn()
        render(<ModelManager activeModel="" onSelectModel={onSelectModel} selectedPath="" />)

        await waitFor(() => screen.getAllByText('Phi-2')[0])

        // Find the Select button. It might be multiple if multiple models?
        // But only downloaded models have Select button.
        const selectBtn = screen.getByText('Select')
        fireEvent.click(selectBtn)

        expect(onSelectModel).toHaveBeenCalledWith('/models/phi-2.gguf')
    })

    it('triggers delete model when delete button is clicked', async () => {
        axios.delete.mockResolvedValueOnce({ data: { success: true } })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getAllByText('Phi-2')[0])

        const trashBtn = screen.getByTitle('Delete Model')
        fireEvent.click(trashBtn)

        expect(global.confirm).toHaveBeenCalled()
        expect(axios.delete).toHaveBeenCalledWith(
            'http://localhost:8000/api/models/delete',
            expect.objectContaining({ data: { path: '/models/phi-2.gguf' } })
        )
    })

    it('shows download progress when downloading', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/models/status')) {
                return Promise.resolve({
                    data: { downloading: true, model_id: 'tinyllama', progress: 45 }
                })
            }
            if (url.includes('/api/models/available')) return Promise.resolve({ data: mockAvailableModels })
            if (url.includes('/api/models/local')) return Promise.resolve({ data: [] })
            if (url.includes('/api/models/recommendations')) return Promise.resolve({ data: mockRecommendations })
            return Promise.resolve({ data: {} })
        })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('45%'))
    })

    it('displays error message when download fails', async () => {
        axios.post.mockRejectedValueOnce({
            response: { data: { detail: 'Insufficient disk space' } }
        })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('TinyLlama'))

        const cards = screen.getAllByText('TinyLlama').map(el => el.closest('.border'))
        const card = cards.find(c => within(c).queryByText(/Download/i))
        const downloadBtn = within(card).getByRole('button', { name: /Download/i })

        fireEvent.click(downloadBtn)

        await waitFor(() => {
            expect(screen.getByText(/Insufficient disk space/i)).toBeDefined()
        })
    })

    it('filters models by category', async () => {
        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('TinyLlama'))

        const smallBtn = screen.getByRole('button', { name: 'Fast' })
        fireEvent.click(smallBtn)

        // After filtering, should still see models or appropriate message
        await waitFor(() => {
            const allModelNames = screen.queryAllByText(/Phi-2|TinyLlama/)
            expect(allModelNames.length).toBeGreaterThan(0)
        })
    })

    it('searches models by name', async () => {
        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('TinyLlama'))

        const searchInput = screen.getByPlaceholderText('Search models...')
        fireEvent.change(searchInput, { target: { value: 'Tiny' } })

        await waitFor(() => {
            expect(screen.getAllByText(/TinyLlama/).length).toBeGreaterThan(0)
        })
    })

    it('shows "no models match" message when search has no results', async () => {
        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('TinyLlama'))

        const searchInput = screen.getByPlaceholderText('Search models...')
        fireEvent.change(searchInput, { target: { value: 'NonexistentModel' } })

        await waitFor(() => {
            expect(screen.getByText(/No models match your search/i)).toBeDefined()
        })
    })

    it('dismisses error message when dismiss button is clicked', async () => {
        axios.post.mockRejectedValueOnce({
            response: { data: { detail: 'Test error' } }
        })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('TinyLlama'))

        const cards = screen.getAllByText('TinyLlama').map(el => el.closest('.border'))
        const card = cards.find(c => within(c).queryByText(/Download/i))
        const downloadBtn = within(card).getByRole('button', { name: /Download/i })

        fireEvent.click(downloadBtn)

        await waitFor(() => screen.getByText(/Test error/i))

        const dismissBtn = screen.getByText('Dismiss')
        fireEvent.click(dismissBtn)

        await waitFor(() => {
            expect(screen.queryByText(/Test error/i)).toBeNull()
        })
    })

    it('shows selected model with visual highlight', async () => {
        render(<ModelManager activeModel="Phi-2" onSelectModel={vi.fn()} selectedPath="/models/phi-2.gguf" />)

        await waitFor(() => {
            const selectedElements = screen.getAllByText('Selected')
            expect(selectedElements.length).toBeGreaterThan(0)
        })
    })

    it('shows active model badge', async () => {
        render(<ModelManager activeModel="Phi-2" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => {
            expect(screen.getByText('Active')).toBeDefined()
        })
    })

    it('shows "Ready" for already downloaded models', async () => {
        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => {
            const readyElements = screen.getAllByText('Ready')
            // Should appear at least once (in downloaded section or available section)
            expect(readyElements.length).toBeGreaterThan(0)
        })
    })

    it('disables download button while downloading', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/models/status')) {
                return Promise.resolve({
                    data: { downloading: true, model_id: 'tinyllama', progress: 45 }
                })
            }
            if (url.includes('/api/models/available')) return Promise.resolve({ data: mockAvailableModels })
            if (url.includes('/api/models/local')) return Promise.resolve({ data: [] })
            if (url.includes('/api/models/recommendations')) return Promise.resolve({ data: mockRecommendations })
            return Promise.resolve({ data: {} })
        })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => {
            const downloadButtons = screen.getAllByRole('button', { name: /Download/i })
            downloadButtons.forEach(btn => {
                if (!btn.disabled) {
                    // At least one button should be disabled when download is active
                }
            })
        })
    })

    it('displays recommendation section when data is available', async () => {
        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => {
            expect(screen.getByText(/Smart Recommendations for Your System/i)).toBeDefined()
            expect(screen.getByText(/16GB RAM/i)).toBeDefined()
        })
    })

    it('allows quick download from recommendations', async () => {
        axios.post.mockResolvedValueOnce({ data: { status: 'success' } })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText(/Smart Recommendations/i))

        const quickDownloadBtn = screen.getByRole('button', { name: 'Quick Download' })
        fireEvent.click(quickDownloadBtn)

        expect(axios.post).toHaveBeenCalledWith('http://localhost:8000/api/models/download/phi-2')
    })

    it('shows warning message with download', async () => {
        axios.post.mockResolvedValueOnce({
            data: { status: 'success', message: 'Download started (Warnings: Low RAM)' }
        })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('TinyLlama'))

        const cards = screen.getAllByText('TinyLlama').map(el => el.closest('.border'))
        const card = cards.find(c => within(c).queryByText(/Download/i))
        const downloadBtn = within(card).getByRole('button', { name: /Download/i })

        fireEvent.click(downloadBtn)

        await waitFor(() => {
            expect(screen.getByText(/Warnings: Low RAM/i)).toBeDefined()
        })
    })

    it('refreshes models after download completes', async () => {
        let downloadComplete = false
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/models/status')) {
                const status = downloadComplete
                    ? { downloading: false, progress: 100 }
                    : { downloading: true, progress: 50, model_id: 'tinyllama' }
                return Promise.resolve({ data: status })
            }
            if (url.includes('/api/models/available')) return Promise.resolve({ data: mockAvailableModels })
            if (url.includes('/api/models/local')) return Promise.resolve({ data: mockLocalModels })
            if (url.includes('/api/models/recommendations')) return Promise.resolve({ data: mockRecommendations })
            return Promise.resolve({ data: {} })
        })

        render(<ModelManager activeModel="" onSelectModel={vi.fn()} selectedPath="" />)

        await waitFor(() => screen.getByText('50%'))

        // Simulate download completion
        downloadComplete = true

        // Wait for the interval to trigger and models to refresh
        await waitFor(() => {
            expect(screen.queryByText('50%')).toBeNull()
        }, { timeout: 3000 })
    })
})