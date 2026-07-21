/**
 * SettingsModal Component Tests
 *
 * Functional tests for SettingsModal using React Testing Library.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render as rtlRender, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { ToastProvider } from '../components/Toast'
import SettingsModal from '../components/SettingsModal'

const render = (ui, options) => rtlRender(
    <ToastProvider>{ui}</ToastProvider>,
    options
)
import axios from 'axios'

// Mock axios and browser globals
vi.mock('axios', () => {
    const mockAxios = {
        get: vi.fn(),
        post: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
        create: vi.fn(),
        interceptors: {
            request: { use: vi.fn() },
            response: { use: vi.fn() },
        },
    }
    mockAxios.create.mockReturnValue(mockAxios)
    return {
        default: mockAxios,
        ...mockAxios
    }
})
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
        axios.get.mockImplementation((url) => {
            if (url.includes('config')) return Promise.resolve({ data: mockConfig })
            if (url.includes('index/status')) return Promise.resolve({ data: { running: false } })
            if (url.includes('folders/history')) return Promise.resolve({ data: mockFolderHistory })
            if (url.includes('cache/stats')) return Promise.resolve({ data: { total_entries: 0, total_hits: 0 } })
            if (url.includes('settings/embeddings')) return Promise.resolve({ data: mockEmbeddingConfig })
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
        render(<SettingsModal isOpen={false} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        expect(screen.queryByRole('dialog')).toBeNull()
    })

    it('closes when Escape key is pressed', async () => {
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        act(() => {
            fireEvent.keyDown(window, { key: 'Escape' })
        })
        expect(onClose).toHaveBeenCalled()
    })

    it('closes when clicking outside the modal', async () => {
        const onClose = vi.fn()
        const { container } = render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        act(() => {
            fireEvent.click(container.firstChild)
        })
        expect(onClose).toHaveBeenCalled()
    })

    it('does not close when clicking inside the modal', async () => {
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        act(() => {
            fireEvent.click(screen.getByText('Settings'))
        })
        expect(onClose).not.toHaveBeenCalled()
    })

    it('shows error toast if API fails to load settings', async () => {
        // Force one of the initialization endpoints to fail
        axios.get.mockImplementation((url) => {
            if (url.includes('folders/history')) return Promise.reject(new Error('Network Error'))
            return Promise.resolve({ data: {} })
        })
        const mockOnClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={mockOnClose} onSave={vi.fn()} activeModel="" />)
        
        await waitFor(() => {
            expect(screen.getByText('Could not load settings')).toBeDefined()
        })
    })

    // ── Data management ─────────────────────────────────────────────────────

    it('clears search history when button is clicked', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        fireEvent.click(screen.getByText('System'))
        await waitFor(() => screen.getByText('Clear history'))
        await act(async () => {
            fireEvent.click(screen.getByText('Clear history'))
        })
        expect(global.confirm).toHaveBeenCalled()
        expect(axios.delete).toHaveBeenCalledWith(expect.stringContaining('search/history'))
    })

    it('removes a folder from the list', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))
        const folderRow = screen.getByText('C:/Users/test/Documents').closest('li')
        await act(async () => {
            fireEvent.click(folderRow.querySelector('button'))
        })
        
        fireEvent.click(screen.getByText('Save Changes'))
        
        await waitFor(() => {
            expect(axios.post).toHaveBeenCalledWith(
                expect.stringContaining('config'),
                expect.objectContaining({ folders: [] })
            )
        })
    })

    it.skip('clears all folder history via dropdown', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        const historyBtn = await screen.findByTitle('Previously indexed folders')
        await act(async () => {
            fireEvent.click(historyBtn)
        })
        await act(async () => {
            fireEvent.click(screen.getByText('Clear All'))
        })
        expect(global.confirm).toHaveBeenCalled()
        expect(axios.delete).toHaveBeenCalledWith(expect.stringContaining('folders/history'))
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
        expect(screen.getByText('Embeddings')).toBeDefined()
    })

    it('expands the Embedding Provider section and shows fields', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('Settings'))
        fireEvent.click(screen.getByText('Embeddings'))
        await waitFor(() => {
            expect(screen.getByText('Local (on-device)')).toBeDefined()
            expect(screen.getByLabelText('Model name')).toBeDefined()
        })
    })

    it('shows API key field only for non-local providers', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))
        fireEvent.click(screen.getByText('Embeddings'))
        await waitFor(() => screen.getByText('Local (on-device)'))

        // Default 'local' → no API key field
        expect(screen.queryByLabelText(/API Key/i)).toBeNull()

        // Switch to huggingface_api → key field appears
        fireEvent.click(screen.getByText('HuggingFace API'))
        await waitFor(() => expect(screen.getByLabelText(/API Key/i)).toBeDefined())
    })

    // ── Save + toast ─────────────────────────────────────────────────────────

    it('POSTs to /api/settings/embeddings when Save is clicked', async () => {
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))

        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            expect(axios.post).toHaveBeenCalledWith(
                expect.stringContaining('settings/embeddings'),
                expect.objectContaining({ provider_type: 'local' })
            )
        })
    })

    it('shows success toast after a successful save', async () => {
        const onClose = vi.fn() // keep modal mounted so we can see toast
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))

        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            expect(screen.getByText('Settings saved')).toBeDefined()
        })
    })

    it('shows error toast when save fails', async () => {
        axios.post.mockRejectedValueOnce({
            response: { data: { detail: 'api_key is required for huggingface_api' } }
        })
        const onClose = vi.fn()
        render(<SettingsModal isOpen={true} onClose={onClose} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))

        fireEvent.click(screen.getByText('Save Changes'))

        await waitFor(() => {
            expect(screen.getByText('api_key is required for huggingface_api')).toBeDefined()
        })
    })

    // ── Add folder ──────────────────────────────────────────────────────────

    it('shows "Validating…" while validating a folder path', async () => {
        // Make validatePath hang for a bit
        let resolveValidate
        axios.post.mockImplementation((url) => {
            if (url.includes('validate-path')) {
                return new Promise((resolve) => { resolveValidate = resolve })
            }
            return Promise.resolve({ data: { status: 'success' } })
        })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))

        const input = screen.getByPlaceholderText(/Documents/i)
        await act(async () => {
            fireEvent.change(input, { target: { value: 'C:/Users/test/NewFolder' } })
        })
        await act(async () => {
            fireEvent.click(screen.getByText('Add folder'))
        })

        // Button should show "Validating…" while waiting
        expect(screen.getByText('Validating…')).toBeDefined()

        // Resolve the validation
        await act(async () => {
            resolveValidate({ data: { valid: true, file_count: 5 } })
        })

        // Button should revert back
        await waitFor(() => {
            expect(screen.getByText('Add folder')).toBeDefined()
        })
    })

    it('adds a valid folder to the list after validation', async () => {
        axios.post.mockImplementation((url, data) => {
            if (url.includes('validate-path')) {
                return Promise.resolve({ data: { valid: true, file_count: 3 } })
            }
            return Promise.resolve({ data: { status: 'success' } })
        })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))

        const input = screen.getByPlaceholderText(/Documents/i)
        await act(async () => {
            fireEvent.change(input, { target: { value: 'C:/Users/test/NewFolder' } })
        })
        await act(async () => {
            fireEvent.click(screen.getByText('Add folder'))
        })

        await waitFor(() => {
            expect(screen.getByText('C:/Users/test/NewFolder')).toBeDefined()
        })
    })

    it('shows error toast for an invalid folder path', async () => {
        axios.post.mockImplementation((url) => {
            if (url.includes('validate-path')) {
                return Promise.resolve({ data: { valid: false, error: 'Path does not exist' } })
            }
            return Promise.resolve({ data: { status: 'success' } })
        })

        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))

        const input = screen.getByPlaceholderText(/Documents/i)
        await act(async () => {
            fireEvent.change(input, { target: { value: 'C:/nonexistent/path' } })
        })
        await act(async () => {
            fireEvent.click(screen.getByText('Add folder'))
        })

        await waitFor(() => {
            expect(screen.getByText('Path does not exist')).toBeDefined()
        })
    })

    it('does not call API when adding an empty folder path', async () => {
        render(<SettingsModal isOpen={true} onClose={vi.fn()} onSave={vi.fn()} activeModel="" />)
        await waitFor(() => screen.getByText('C:/Users/test/Documents'))

        // Input is empty by default, click Add folder
        await act(async () => {
            fireEvent.click(screen.getByText('Add folder'))
        })

        // validatePath should NOT have been called
        const validateCalls = axios.post.mock.calls.filter(c => c[0]?.includes('validate-path'))
        expect(validateCalls.length).toBe(0)
    })
})
