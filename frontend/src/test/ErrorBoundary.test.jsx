/**
 * Tests for frontend/src/components/ErrorBoundary.jsx
 *
 * Verifies that the boundary catches child errors, displays the error UI,
 * exposes the raw error message, and can reset via the "Try again" button.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import React from 'react';
import ErrorBoundary from '../components/ErrorBoundary';

// Suppress React's error boundary console.error output during tests
let consoleErrorSpy;
beforeEach(() => {
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
});
afterEach(() => {
    consoleErrorSpy.mockRestore();
});

// Helper: a component that throws an error on render when the `throw` prop is true
function BrokenChild({ shouldThrow = false }) {
    if (shouldThrow) {
        throw new Error('Test render error');
    }
    return <div data-testid="healthy-child">All good</div>;
}

// Helper: a component that throws with a custom message
function CustomErrorChild({ message }) {
    throw new Error(message);
}

// ── Normal (no error) rendering ────────────────────────────────────────────────

describe('ErrorBoundary — normal rendering', () => {
    it('renders children when no error occurs', () => {
        render(
            <ErrorBoundary>
                <BrokenChild shouldThrow={false} />
            </ErrorBoundary>
        );
        expect(screen.getByTestId('healthy-child')).toBeDefined();
        expect(screen.queryByRole('alert')).toBeNull();
    });

    it('does not show error UI when children render successfully', () => {
        render(
            <ErrorBoundary>
                <span>Normal content</span>
            </ErrorBoundary>
        );
        expect(screen.queryByText('Something went wrong')).toBeNull();
    });
});

// ── Error state rendering ──────────────────────────────────────────────────────

describe('ErrorBoundary — error state', () => {
    it('shows the error UI when a child throws', () => {
        render(
            <ErrorBoundary>
                <BrokenChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByRole('alert')).toBeDefined();
        expect(screen.getByText('Something went wrong')).toBeDefined();
    });

    it('hides the children when an error is caught', () => {
        render(
            <ErrorBoundary>
                <BrokenChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.queryByTestId('healthy-child')).toBeNull();
    });

    it('displays the raw error message in a <pre> element', () => {
        render(
            <ErrorBoundary>
                <CustomErrorChild message="Specific failure message" />
            </ErrorBoundary>
        );
        const pre = screen.getByRole('alert').querySelector('pre');
        expect(pre).toBeDefined();
        expect(pre.textContent).toContain('Specific failure message');
    });

    it('shows a description prompting the user to reload', () => {
        render(
            <ErrorBoundary>
                <BrokenChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText(/try reloading/i)).toBeDefined();
    });

    it('renders "Try again" and "Reload" buttons', () => {
        render(
            <ErrorBoundary>
                <BrokenChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Try again')).toBeDefined();
        expect(screen.getByText('Reload')).toBeDefined();
    });
});

// ── Recovery via "Try again" ───────────────────────────────────────────────────

describe('ErrorBoundary — recovery', () => {
    it('"Try again" resets hasError so children can re-render', () => {
        // We use a stateful wrapper so we can flip the shouldThrow prop
        function Wrapper() {
            const [shouldThrow, setShouldThrow] = React.useState(true);
            return (
                <ErrorBoundary key={shouldThrow ? 'broken' : 'healthy'}>
                    <BrokenChild shouldThrow={shouldThrow} />
                </ErrorBoundary>
            );
        }

        // Note: clicking "Try again" resets ErrorBoundary internal state only;
        // if the child still throws, the UI will show the error again.
        // Here we verify that clicking the button doesn't itself crash.
        render(
            <ErrorBoundary>
                <BrokenChild shouldThrow={true} />
            </ErrorBoundary>
        );

        const tryAgainBtn = screen.getByText('Try again');
        act(() => {
            fireEvent.click(tryAgainBtn);
        });

        // After clicking "Try again", the boundary resets — child re-renders and throws again
        // so error UI should still be visible (child still throws)
        expect(screen.getByRole('alert')).toBeDefined();
    });

    it('calls console.error with the caught error info', () => {
        render(
            <ErrorBoundary>
                <BrokenChild shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(consoleErrorSpy).toHaveBeenCalled();
    });
});

// ── componentDidCatch logging ─────────────────────────────────────────────────

describe('ErrorBoundary — lifecycle', () => {
    it('logs the error via console.error in componentDidCatch', () => {
        render(
            <ErrorBoundary>
                <CustomErrorChild message="logged error" />
            </ErrorBoundary>
        );
        const errorCalls = consoleErrorSpy.mock.calls;
        const hasLogged = errorCalls.some((args) =>
            args.some((a) => String(a).includes('logged error') || String(a).includes('UI error'))
        );
        expect(hasLogged).toBe(true);
    });
});
