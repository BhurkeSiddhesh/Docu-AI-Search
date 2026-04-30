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
        return await screen.findByText('Library', {}, { timeout: 8000 })
    }

    it('renders correctly when open', async () => {
        await openModal()
        expect(screen.getByText('System Configuration')).toBeDefined()
        expect(screen.getByText('C:/Users/test/Documents')).toBeDefined()
    })

    it('switches between sections', async () => {
        await openModal()
        
        // Cloud AI
        fireEvent.click(screen.getByText('Cloud AI'))
        await screen.findByText('Cloud Intelligence', {}, { timeout: 8000 })

        // Local LLM
        fireEvent.click(screen.getByText('Local LLM'))
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
        const rebuildBtn = screen.getByText('Rebuild Index')
        fireEvent.click(rebuildBtn)
        await waitFor(() => {
            expect(axios.post).toHaveBeenCalled()
        })
        await screen.findByText('Index rebuild started')
    }, 15000)

    it('clears AI response cache when button is clicked', async () => {
        await openModal()
        fireEvent.click(screen.getByText('System'))
        await screen.findByText('System Hygiene', {}, { timeout: 8000 })
        
        const purgeBtn = screen.getByText('Purge AI Cache')
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
        await waitFor(() => screen.getByText('System Configuration'))

        fireEvent.click(screen.getByText('Library'))

        await waitFor(() => {
            expect(screen.getByText(/42 entries/)).toBeDefined()
            expect(screen.getByText(/128 hits saved/)).toBeDefined()
        })
    })

    it('changes model name in embedding config', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('System Configuration'))

        fireEvent.click(screen.getByText('System'))
        await waitFor(() => screen.getByLabelText('Model Name / Repo ID'))

        const modelInput = screen.getByLabelText('Model Name / Repo ID')
        fireEvent.change(modelInput, { target: { value: 'new-model-name' } })

        expect(modelInput.value).toBe('new-model-name')
    })

    it('disables save button while saving', async () => {
        // Make save take longer
        axios.post.mockImplementation(() => new Promise(resolve => setTimeout(() => resolve({ data: { status: 'success' } }), 100)))

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('System Configuration'))

        const saveButton = screen.getByText('Save Changes')
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
        await waitFor(() => screen.getByText('System Configuration'))

        fireEvent.click(screen.getByText('Cloud AI'))
        await waitFor(() => screen.getByLabelText(/API Key/i))

        // API key should show placeholder
        const apiKeyInput = screen.getByLabelText(/API Key/i)
        expect(apiKeyInput.value).toBe('••••••••')

        // Save without changing the API key
        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            // Should not send the placeholder
            const embeddingCall = axios.post.mock.calls.find(call =>
                call[0] === 'http://localhost:8000/api/settings/embeddings'
            )
            expect(embeddingCall[1].api_key).toBeUndefined()
        })
    })

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
        await waitFor(() => screen.getByText('System Configuration'))

        // Click add folder (should not crash)
        fireEvent.click(screen.getByText('Add Folder'))

        // Should handle error gracefully (no crash)
        await waitFor(() => expect(axios.get).toHaveBeenCalled())
    })

    it('updates API keys for different providers', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('System Configuration'))

        fireEvent.click(screen.getByText('Cloud AI'))
        await waitFor(() => screen.getByText('OpenAI (ChatGPT)'))

        // Find all API key inputs
        const inputs = screen.getAllByPlaceholderText(/sk-|AIza|xai-/i)
        expect(inputs.length).toBeGreaterThan(0)

        // Change first one
        fireEvent.change(inputs[0], { target: { value: 'new-api-key' } })
        expect(inputs[0].value).toBe('new-api-key')
    })

    it('shows folder history dropdown when Recent History button is clicked', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('System Configuration'))

        const historyButton = screen.getByText('Recent History')
        fireEvent.click(historyButton)

        await waitFor(() => {
            expect(screen.getByText('Previously Indexed')).toBeDefined()
            expect(screen.getByText('C:/Old/Folder')).toBeDefined()
        })
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
        await waitFor(() => screen.getByText('System Configuration'))

        fireEvent.click(screen.getByText('Recent History'))

        await waitFor(() => {
            expect(screen.getByText('No indexed folders yet.')).toBeDefined()
        })
    })

    it('dismisses toast after 3 seconds', async () => {
        // shouldAdvanceTime keeps real time flowing so waitFor polling works,
        // while still letting us control the fake clock for the toast timer.
        vi.useFakeTimers({ shouldAdvanceTime: true })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('System Configuration'))

        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            expect(screen.getByText('Settings saved successfully!')).toBeDefined()
        })

        // Advance the fake clock past the 3 s auto-dismiss threshold
        act(() => { vi.advanceTimersByTime(3000) })

        // Poll for the DOM update instead of asserting synchronously — React
        // may batch the state update and re-render asynchronously
        await waitFor(() => {
            expect(screen.queryByText('Settings saved successfully!')).toBeNull()
        })

        vi.useRealTimers()
    })
})
})