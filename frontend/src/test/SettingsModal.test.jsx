/**
 * SettingsModal Component Tests
 *
 * Functional tests for SettingsModal using React Testing Library.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act, cleanup } from '@testing-library/react'
import SettingsModal from '../components/SettingsModal'
import axios from 'axios'

// Mock axios correctly
vi.mock('axios')

// Mock ModelManager
vi.mock('../components/ModelManager', () => ({
    default: () => <div data-testid="model-manager">Model Manager Mock</div>
}))

const mockConfig = {
    folders: ['C:/Users/test/Documents'],
    auto_index: false,
    provider: 'local',
    openai_api_key_set: false,
    gemini_api_key_set: false,
    anthropic_api_key_set: false,
    grok_api_key_set: false,
    local_model_path: '',
    local_model_type: 'llamacpp'
}

const mockEmbeddingConfig = {
    provider_type: 'local',
    model_name: 'Alibaba-NLP/gte-Qwen2-1.5B-instruct',
    api_key_set: false,
}

const mockFolderHistory = ['C:/Old/Folder']

describe('SettingsModal Component', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        
        const mockFn = vi.fn(() => true)
        global.confirm = mockFn
        global.alert = vi.fn()
        if (typeof window !== 'undefined') {
            window.confirm = mockFn
            window.alert = global.alert
        }

        axios.get.mockImplementation((url) => {
            if (url.includes('/api/config')) return Promise.resolve({ data: mockConfig })
            if (url.includes('/api/index/status')) return Promise.resolve({ data: { running: false } })
            if (url.includes('/api/folders/history')) return Promise.resolve({ data: mockFolderHistory })
            if (url.includes('/api/cache/stats')) return Promise.resolve({ data: { total_entries: 0, total_hits: 0 } })
            if (url.includes('/api/settings/embeddings')) return Promise.resolve({ data: mockEmbeddingConfig })
            return Promise.resolve({ data: {} })
        })
        axios.post.mockResolvedValue({ data: { status: 'success' } })
        axios.delete.mockResolvedValue({ data: { success: true } })
    })

    afterEach(() => {
        cleanup()
    })

    const openModal = async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        return await screen.findAllByText('Library', {}, { timeout: 8000 })
    }

    it('renders correctly when open', async () => {
        await openModal()
        expect(screen.getByText('System Configuration')).toBeDefined()
        expect(screen.getByText('C:/Users/test/Documents')).toBeDefined()
    })

    it('switches between sections', async () => {
        await openModal()
        
        // Cloud AI
        fireEvent.click(screen.getAllByText('Cloud AI')[0])
        await screen.findByText('Cloud Intelligence', {}, { timeout: 8000 })

        // Local LLM
        fireEvent.click(screen.getAllByText('Local LLM')[0])
        await screen.findByText('Model Manager Mock', {}, { timeout: 8000 })
    }, 20000)

    it('removes a folder', async () => {
        await openModal()
        const removeBtn = screen.getByLabelText('Remove C:/Users/test/Documents from index')
        fireEvent.click(removeBtn)
        await waitFor(() => {
            expect(screen.queryByText('C:/Users/test/Documents')).toBeNull()
        })
    })

    it('triggers rebuild index', async () => {
        await openModal()
        const rebuildBtn = screen.getAllByText('Rebuild Index')[0]
        fireEvent.click(rebuildBtn)
        await waitFor(() => {
            expect(axios.post).toHaveBeenCalled()
        })
        await screen.findByText('Index rebuild started')
    }, 15000)

    it('clears AI response cache when button is clicked', async () => {
        await openModal()
        fireEvent.click(screen.getAllByText('System')[0])
        await screen.findByText('System Hygiene', {}, { timeout: 8000 })
        
        const purgeBtn = screen.getAllByText('Purge AI Cache')[0]
        fireEvent.click(purgeBtn)
        
        await waitFor(() => {
            expect(axios.post).toHaveBeenCalledWith(expect.stringContaining('/api/cache/clear'))
        })
    })

    it('displays cache statistics', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/config')) return Promise.resolve({ data: mockConfig })
            if (url.includes('/api/index/status')) return Promise.resolve({ data: { running: false } })
            if (url.includes('/api/folders/history')) return Promise.resolve({ data: mockFolderHistory })
            if (url.includes('/api/cache/stats')) return Promise.resolve({
                data: { total_entries: 42, total_hits: 128 }
            })
            if (url.includes('/api/settings/embeddings')) return Promise.resolve({ data: mockEmbeddingConfig })
            return Promise.resolve({ data: {} })
        })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        fireEvent.click(screen.getAllByText('System')[0])

        await waitFor(() => {
            expect(screen.getAllByText(/42/)[0]).toBeDefined()
            expect(screen.getAllByText(/128/)[0]).toBeDefined()
        })
    }, 20000)

    it('changes model name in embedding config', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        fireEvent.click(screen.getAllByText('Embeddings')[0])
        await screen.findByLabelText('Model Architecture', {}, { timeout: 8000 })

        const modelInput = screen.getByLabelText('Model Architecture')
        fireEvent.change(modelInput, { target: { value: 'new-model-name' } })

        expect(modelInput.value).toBe('new-model-name')
    })

    it('disables save button while saving', async () => {
        // Make save take longer
        axios.post.mockImplementation(() => new Promise(resolve => setTimeout(() => resolve({ data: { status: 'success' } }), 100)))

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        const saveButton = screen.getAllByText('Apply Changes')[0]
        fireEvent.click(saveButton)

        // Button should be disabled during save
        expect(saveButton.closest('button').disabled).toBe(true)
    })

    it('does not send placeholder API key when saving', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/config')) return Promise.resolve({ data: mockConfig })
            if (url.includes('/api/index/status')) return Promise.resolve({ data: { running: false } })
            if (url.includes('/api/folders/history')) return Promise.resolve({ data: mockFolderHistory })
            if (url.includes('/api/cache/stats')) return Promise.resolve({ data: { total_entries: 0, total_hits: 0 } })
            if (url.includes('/api/settings/embeddings')) return Promise.resolve({
                data: { provider_type: 'commercial_api', model_name: 'test', api_key_set: true }
            })
            return Promise.resolve({ data: {} })
        })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        fireEvent.click(screen.getAllByText('Embeddings')[0])
        await screen.findByLabelText(/Embedding API Key/i, {}, { timeout: 8000 })

        // API key should show placeholder
        const apiKeyInput = screen.getByLabelText(/Embedding API Key/i)
        expect(apiKeyInput.value).toBe('••••••••')

        // Save without changing the API key
        fireEvent.click(screen.getAllByText('Apply Changes')[0])

        await waitFor(() => {
            // Should not send the placeholder
            const embeddingCall = axios.post.mock.calls.find(call =>
                call[0] === 'http://localhost:8000/api/settings/embeddings'
            )
            expect(embeddingCall[1].api_key).toBeUndefined()
        })
    }, 20000)

    it('handles browse folder error gracefully', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/browse')) return Promise.reject(new Error('Failed to open dialog'))
            if (url.includes('/api/config')) return Promise.resolve({ data: mockConfig })
            if (url.includes('/api/index/status')) return Promise.resolve({ data: { running: false } })
            if (url.includes('/api/folders/history')) return Promise.resolve({ data: mockFolderHistory })
            if (url.includes('/api/cache/stats')) return Promise.resolve({ data: { total_entries: 0, total_hits: 0 } })
            if (url.includes('/api/settings/embeddings')) return Promise.resolve({ data: mockEmbeddingConfig })
            return Promise.resolve({ data: {} })
        })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        // Click add folder (should not crash)
        fireEvent.click(screen.getAllByText('Add Folder')[0])

        // Should handle error gracefully (no crash)
        await waitFor(() => expect(axios.get).toHaveBeenCalled())
    })

    it('updates API keys for different providers', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        fireEvent.click(screen.getAllByText('Cloud AI')[0])
        await screen.findByText('Cloud Intelligence', {}, { timeout: 8000 })

        // Find all API key inputs
        const inputs = screen.getAllByPlaceholderText(/Enter API Key.../i)
        expect(inputs.length).toBeGreaterThan(0)

        // Change first one
        fireEvent.change(inputs[0], { target: { value: 'new-api-key' } })
        expect(inputs[0].value).toBe('new-api-key')
    })

    it('shows folder history dropdown when Recent History button is clicked', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        const historyButton = screen.getByLabelText(/history/i)
        fireEvent.click(historyButton)

        await screen.findByText('Previously Indexed', {}, { timeout: 8000 })
        expect(screen.getAllByText('C:/Old/Folder')[0]).toBeDefined()
    })

    it('handles empty folder history', async () => {
        axios.get.mockImplementation((url) => {
            if (url.includes('/api/config')) return Promise.resolve({ data: mockConfig })
            if (url.includes('/api/index/status')) return Promise.resolve({ data: { running: false } })
            if (url.includes('/api/folders/history')) return Promise.resolve({ data: [] })  // Empty
            if (url.includes('/api/cache/stats')) return Promise.resolve({ data: { total_entries: 0, total_hits: 0 } })
            if (url.includes('/api/settings/embeddings')) return Promise.resolve({ data: mockEmbeddingConfig })
            return Promise.resolve({ data: {} })
        })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        fireEvent.click(screen.getByLabelText(/history/i))

        await screen.findByText('No indexed folders yet.', {}, { timeout: 8000 })
    })

    it('dismisses toast when the X button is clicked', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await screen.findByText('System Configuration', {}, { timeout: 8000 })

        fireEvent.click(screen.getAllByText('Apply Changes')[0])

        await screen.findByText('Configuration updated!', {}, { timeout: 8000 })

        // Click the dismiss button — same onDismiss callback the 3 s timer uses
        fireEvent.click(screen.getByRole('button', { name: /dismiss notification/i }))

        await waitFor(() => {
            expect(screen.queryByText('Configuration updated!')).toBeNull()
        })
    })
})