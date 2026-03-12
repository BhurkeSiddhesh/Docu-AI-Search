/**
 * SettingsModal Component Tests
 *
 * Functional tests for SettingsModal using React Testing Library.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import SettingsModal from '../components/SettingsModal'
import axios from 'axios'

// Mock axios and browser globals
vi.mock('axios')
global.confirm = vi.fn(() => true)
global.alert = vi.fn()

// Mock ModelManager since it is tested separately
vi.mock('../components/ModelManager', () => ({
    default: () => <div data-testid="model-manager">Model Manager Mock</div>
}))

const mockConfig = {
    folders: ['C:/Users/test/Documents'],
    auto_index: false,
    provider: 'local',
    openai_api_key: '',
    gemini_api_key: '',
    anthropic_api_key: '',
    grok_api_key: '',
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

    // ── Rendering ────────────────────────────────────────────────────────────

    it('renders correctly when open', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => {
            expect(screen.getByText('Settings')).toBeDefined()
            expect(screen.getByText('C:/Users/test/Documents')).toBeDefined()
        })
    })

    it('does not render when closed', () => {
        const { container } = render(<SettingsModal isOpen={false} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        expect(container.firstChild).toBeNull()
    })

    it('closes when Escape key is pressed', () => {
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        fireEvent.keyDown(window, { key: 'Escape' })
        expect(onClose).toHaveBeenCalled()
    })

    it('closes when clicking outside the modal', () => {
        const onClose = vi.fn()
        const { container } = render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        fireEvent.click(container.firstChild)
        expect(onClose).toHaveBeenCalled()
    })

    it('does not close when clicking inside the modal', () => {
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        fireEvent.click(screen.getByText('Settings'))
        expect(onClose).not.toHaveBeenCalled()
    })

    // ── Data management ─────────────────────────────────────────────────────

    it('clears search history when button is clicked', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        fireEvent.click(screen.getByText('Clear Search History'))
        expect(global.confirm).toHaveBeenCalled()
        expect(axios.delete).toHaveBeenCalledWith('http://localhost:8000/api/search/history')
    })

    it('removes a folder from the list', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))
        const folderRow = screen.getByText('C:/Users/test/Documents').closest('div').parentElement
        fireEvent.click(folderRow.querySelector('button'))
        expect(axios.post).toHaveBeenCalledWith(
            'http://localhost:8000/api/config',
            expect.objectContaining({ folders: [] })
        )
    })

    it('clears all folder history via dropdown', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        const historyBtn = await screen.findByTitle('Previously indexed folders')
        fireEvent.click(historyBtn)
        fireEvent.click(screen.getByText('Clear All'))
        expect(global.confirm).toHaveBeenCalled()
        expect(axios.delete).toHaveBeenCalledWith('http://localhost:8000/api/folders/history')
    })

    // ── Accessibility ────────────────────────────────────────────────────────

    it('has accessible labels', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))
        expect(screen.getByLabelText('Close settings')).toBeDefined()
        expect(screen.getByLabelText('Remove C:/Users/test/Documents from index')).toBeDefined()
    })

    // ── Embedding Provider section ───────────────────────────────────────────

    it('renders the Embedding Provider toggle button', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        expect(screen.getByText('Embedding Provider')).toBeDefined()
    })

    it('expands the Embedding Provider section and shows fields', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        fireEvent.click(screen.getByText('Embedding Provider'))
        await waitFor(() => {
            expect(screen.getByLabelText('Provider Type')).toBeDefined()
            expect(screen.getByLabelText('Model Name / Repo ID')).toBeDefined()
        })
    })

    it('shows API key field only for non-local providers', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        fireEvent.click(screen.getByText('Embedding Provider'))
        await waitFor(() => screen.getByLabelText('Provider Type'))

        // Default 'local' → no API key field
        expect(screen.queryByLabelText(/API Key/i)).toBeNull()

        // Switch to huggingface_api → key field appears
        fireEvent.change(screen.getByLabelText('Provider Type'), { target: { value: 'huggingface_api' } })
        await waitFor(() => expect(screen.getByLabelText(/API Key/i)).toBeDefined())
    })

    // ── Save + toast ─────────────────────────────────────────────────────────

    it('POSTs to /api/settings/embeddings when Save is clicked', async () => {
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))

        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            expect(axios.post).toHaveBeenCalledWith(
                'http://localhost:8000/api/settings/embeddings',
                expect.objectContaining({ provider_type: 'local' })
            )
        })
    })

    it('shows success toast after a successful save', async () => {
        const onClose = vi.fn() // keep modal mounted so we can see toast
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))

        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            expect(screen.getByText('Settings saved successfully!')).toBeDefined()
        })
    })

    it('shows error toast when save fails', async () => {
        axios.post.mockRejectedValueOnce({
            response: { data: { detail: 'api_key is required for huggingface_api' } }
        })
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))

        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            expect(screen.getByText('api_key is required for huggingface_api')).toBeDefined()
        })
    })
})
