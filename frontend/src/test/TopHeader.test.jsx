/**
 * TopHeader Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import TopHeader from '../components/TopHeader';

describe('TopHeader Component', () => {
    it('renders without crashing', () => {
        render(<TopHeader title="Dashboard" />);
    });

    it('displays the provided title', () => {
        render(<TopHeader title="Knowledge Base" />);
        expect(screen.getByText('Knowledge Base')).toBeDefined();
    });

    it('renders the Neural Link Active status indicator', () => {
        render(<TopHeader title="Dashboard" />);
        expect(screen.getByText(/Neural Link Active/i)).toBeDefined();
    });

    it('renders the hamburger menu button', () => {
        render(<TopHeader title="Dashboard" onMenuOpen={vi.fn()} />);
        expect(screen.getByLabelText('Open navigation menu')).toBeDefined();
    });

    it('calls onMenuOpen when the hamburger button is clicked', () => {
        const onMenuOpen = vi.fn();
        render(<TopHeader title="Dashboard" onMenuOpen={onMenuOpen} />);

        fireEvent.click(screen.getByLabelText('Open navigation menu'));

        expect(onMenuOpen).toHaveBeenCalledOnce();
    });

    it('renders with different title values', () => {
        const { rerender } = render(<TopHeader title="Telemetry" />);
        expect(screen.getByText('Telemetry')).toBeDefined();

        rerender(<TopHeader title="Neural Search" />);
        expect(screen.getByText('Neural Search')).toBeDefined();
    });

    it('does not crash when onMenuOpen is not provided', () => {
        render(<TopHeader title="Dashboard" />);
        expect(() =>
            fireEvent.click(screen.getByLabelText('Open navigation menu'))
        ).not.toThrow();
    });
});
