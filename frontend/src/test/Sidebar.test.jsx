import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import Sidebar from '../components/Sidebar';

describe('Sidebar Component', () => {
    const defaultProps = {
        activeTab: 'search',
        onTabChange: vi.fn(),
        onOpenSettings: vi.fn(),
        onOpenHistory: vi.fn(),
        onNewSearch: vi.fn(),
        isOpen: true,
        onClose: vi.fn(),
        darkMode: false,
        onToggleDark: vi.fn(),
    };

    it('renders workspace navigation tabs correctly and omits portfolio tab', () => {
        render(<Sidebar {...defaultProps} />);

        expect(screen.getByText('Search')).toBeDefined();
        expect(screen.getByText('Library')).toBeDefined();
        expect(screen.getByText('Graph')).toBeDefined();
        expect(screen.getByText('Benchmarks')).toBeDefined();

        // Ensure Portfolio tab is completely dropped
        expect(screen.queryByText('Portfolio')).toBeNull();
        expect(screen.queryByText('portfolio')).toBeNull();
    });

    it('handles tab changes when clicking navigation items', () => {
        const onTabChange = vi.fn();
        const onClose = vi.fn();
        render(<Sidebar {...defaultProps} onTabChange={onTabChange} onClose={onClose} />);

        fireEvent.click(screen.getByText('Library'));
        expect(onTabChange).toHaveBeenCalledWith('library');
        expect(onClose).toHaveBeenCalled();

        fireEvent.click(screen.getByText('Graph'));
        expect(onTabChange).toHaveBeenCalledWith('graph');

        fireEvent.click(screen.getByText('Benchmarks'));
        expect(onTabChange).toHaveBeenCalledWith('benchmarks');
    });

    it('triggers action buttons for New Search, History, Settings, and Dark Mode', () => {
        const onNewSearch = vi.fn();
        const onOpenHistory = vi.fn();
        const onOpenSettings = vi.fn();
        const onToggleDark = vi.fn();

        render(
            <Sidebar
                {...defaultProps}
                onNewSearch={onNewSearch}
                onOpenHistory={onOpenHistory}
                onOpenSettings={onOpenSettings}
                onToggleDark={onToggleDark}
            />
        );

        fireEvent.click(screen.getByRole('button', { name: /new search/i }));
        expect(onNewSearch).toHaveBeenCalled();

        fireEvent.click(screen.getByRole('button', { name: /history/i }));
        expect(onOpenHistory).toHaveBeenCalled();

        fireEvent.click(screen.getByRole('button', { name: /settings/i }));
        expect(onOpenSettings).toHaveBeenCalled();

        fireEvent.click(screen.getByRole('button', { name: /dark mode/i }));
        expect(onToggleDark).toHaveBeenCalled();
    });
});
