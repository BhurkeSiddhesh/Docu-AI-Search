/**
 * SideNavBar Component Tests
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import SideNavBar from '../components/SideNavBar';

const defaultProps = {
    activeTab: 'dashboard',
    setActiveTab: vi.fn(),
    onNewSearch: vi.fn(),
    setIsSettingsOpen: vi.fn(),
    setIsHistoryOpen: vi.fn(),
    isOpen: true,
    onClose: vi.fn(),
};

describe('SideNavBar Component', () => {
    it('renders the brand name', () => {
        render(<SideNavBar {...defaultProps} />);
        expect(screen.getByText(/Docu/i)).toBeDefined();
        expect(screen.getByText(/AI/i)).toBeDefined();
    });

    it('renders all primary nav items', () => {
        render(<SideNavBar {...defaultProps} />);
        expect(screen.getByText('Neural Search')).toBeDefined();
        expect(screen.getByText('Knowledge Base')).toBeDefined();
        expect(screen.getByText('Telemetry')).toBeDefined();
    });

    it('calls setActiveTab when a nav item is clicked', () => {
        const setActiveTab = vi.fn();
        render(<SideNavBar {...defaultProps} setActiveTab={setActiveTab} />);

        fireEvent.click(screen.getByText('Knowledge Base'));

        expect(setActiveTab).toHaveBeenCalledWith('library');
    });

    it('calls onClose when a nav item is clicked', () => {
        const onClose = vi.fn();
        render(<SideNavBar {...defaultProps} onClose={onClose} />);

        fireEvent.click(screen.getByText('Telemetry'));

        expect(onClose).toHaveBeenCalled();
    });

    it('calls onClose when the close button is clicked', () => {
        const onClose = vi.fn();
        render(<SideNavBar {...defaultProps} onClose={onClose} />);

        fireEvent.click(screen.getByLabelText('Close menu'));

        expect(onClose).toHaveBeenCalled();
    });

    it('shows mobile backdrop when isOpen is true', () => {
        const { container } = render(<SideNavBar {...defaultProps} isOpen={true} />);
        const backdrop = container.querySelector('.bg-black\\/40');
        expect(backdrop).not.toBeNull();
    });

    it('hides mobile backdrop when isOpen is false', () => {
        const { container } = render(<SideNavBar {...defaultProps} isOpen={false} />);
        const backdrop = container.querySelector('.bg-black\\/40');
        expect(backdrop).toBeNull();
    });

    it('calls onClose when backdrop is clicked', () => {
        const onClose = vi.fn();
        const { container } = render(<SideNavBar {...defaultProps} onClose={onClose} isOpen={true} />);

        const backdrop = container.querySelector('.bg-black\\/40');
        fireEvent.click(backdrop);

        expect(onClose).toHaveBeenCalled();
    });

    it('does not crash when onClose is undefined', () => {
        const propsWithoutClose = { ...defaultProps, onClose: undefined };
        expect(() => {
            render(<SideNavBar {...propsWithoutClose} />);
            fireEvent.click(screen.getByText('Neural Search'));
        }).not.toThrow();
    });
});
