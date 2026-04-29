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
        await screen.findByText('System Engine', {}, { timeout: 8000 })
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
        await screen.findByText('Cache Purged')
    }, 15000)
})