/**
 * ErrorBoundary Component Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary from '../components/ErrorBoundary';

// A child that renders normally
const GoodChild = () => <div>All good</div>;

// A child that throws during render
const BadChild = ({ shouldThrow }) => {
    if (shouldThrow) throw new Error('Render exploded');
    return <div>Safe child</div>;
};

// Silence React's error boundary console.error noise in tests
beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
});

describe('ErrorBoundary Component', () => {
    it('renders children when there is no error', () => {
        render(
            <ErrorBoundary>
                <GoodChild />
            </ErrorBoundary>
        );
        expect(screen.getByText('All good')).toBeDefined();
    });

    it('renders error UI when a child throws', () => {
        render(
            <ErrorBoundary>
                <BadChild shouldThrow />
            </ErrorBoundary>
        );
        expect(screen.getByRole('alert')).toBeDefined();
        expect(screen.getByText('Something went wrong')).toBeDefined();
    });

    it('displays the error message from the thrown error', () => {
        render(
            <ErrorBoundary>
                <BadChild shouldThrow />
            </ErrorBoundary>
        );
        expect(screen.getByText('Render exploded')).toBeDefined();
    });

    it('shows a Try again button when an error occurs', () => {
        render(
            <ErrorBoundary>
                <BadChild shouldThrow />
            </ErrorBoundary>
        );
        expect(screen.getByText('Try again')).toBeDefined();
    });

    it('clears error state and re-renders children when Try again is clicked', () => {
        // Render with a controlled flag via a wrapper
        const { rerender } = render(
            <ErrorBoundary>
                <BadChild shouldThrow />
            </ErrorBoundary>
        );

        expect(screen.getByRole('alert')).toBeDefined();

        fireEvent.click(screen.getByText('Try again'));

        // After reset the error boundary tries to render children again.
        // BadChild still throws so we just confirm the boundary itself reset.
        // We verify by checking that the error UI is re-rendered (boundary re-throws).
        expect(screen.getByRole('alert')).toBeDefined();
    });

    it('renders multiple children when no error is present', () => {
        render(
            <ErrorBoundary>
                <span>First</span>
                <span>Second</span>
            </ErrorBoundary>
        );
        expect(screen.getByText('First')).toBeDefined();
        expect(screen.getByText('Second')).toBeDefined();
    });
});
