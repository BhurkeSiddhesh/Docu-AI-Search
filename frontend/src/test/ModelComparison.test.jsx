import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, act } from '@testing-library/react'
import ModelComparison from '../components/ModelComparison'
import axios from 'axios'

// Mock axios
vi.mock('axios')

describe('ModelComparison Component', () => {
    const mockModels = [
        { name: 'Model A', path: '/models/model-a' },
        { name: 'Model B', path: '/models/model-b' },
        { name: 'Model C', path: '/models/model-c' }
    ]

    beforeEach(() => {
        vi.clearAllMocks()
        axios.get.mockResolvedValue({ data: mockModels })
    })

    it('renders model selection dropdowns correctly', async () => {
        await act(async () => {
            render(<ModelComparison />)
        })

        // Wait for models to load
        await waitFor(() => {
            expect(axios.get).toHaveBeenCalledWith('http://localhost:8000/api/models/local')
        })

        // Check if dropdowns are present
        const dropdowns = screen.getAllByRole('combobox')
        expect(dropdowns).toHaveLength(2)

        // Check options in the first dropdown
        // Note: We can't easily check the 'key' prop directly in the DOM,
        // but we can verify the options are rendered correctly.
        const options = screen.getAllByRole('option')
        // 3 options per dropdown * 2 dropdowns = 6 options
        expect(options).toHaveLength(6)

        expect(options[0].value).toBe('/models/model-a')
        expect(options[0].textContent).toBe('Model A')
        expect(options[1].value).toBe('/models/model-b')
        expect(options[1].textContent).toBe('Model B')
    })
})
