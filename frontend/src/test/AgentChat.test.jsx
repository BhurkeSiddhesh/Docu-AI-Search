/**
 * AgentChat Component Tests
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import AgentChat from '../components/AgentChat';

// EventSource must be a constructor (class), not a plain arrow function.
class MockEventSource {
    constructor(url) {
        MockEventSource.lastInstance = this;
        MockEventSource.lastUrl = url;
        this.onmessage = null;
        this.onerror = null;
        this.close = vi.fn();
    }
}
MockEventSource.lastInstance = null;
MockEventSource.lastUrl = null;

beforeEach(() => {
    MockEventSource.lastInstance = null;
    MockEventSource.lastUrl = null;
    vi.stubGlobal('EventSource', MockEventSource);
});

afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
});

describe('AgentChat Component', () => {
    it('does not create an EventSource when query is empty', () => {
        render(<AgentChat query="" />);
        expect(MockEventSource.lastInstance).toBeNull();
    });

    it('creates an EventSource when a query is provided', () => {
        render(<AgentChat query="What is machine learning?" />);
        expect(MockEventSource.lastInstance).not.toBeNull();
        expect(MockEventSource.lastUrl).toContain('What%20is%20machine%20learning%3F');
    });

    it('renders a thought event received from the stream', async () => {
        render(<AgentChat query="test" />);
        const instance = MockEventSource.lastInstance;

        await act(async () => {
            instance.onmessage({
                data: JSON.stringify({ type: 'thought', content: 'Thinking deeply...' }),
            });
        });

        expect(screen.getByText('Thinking deeply...')).toBeDefined();
    });

    it('renders an answer event and closes the stream', async () => {
        render(<AgentChat query="test" />);
        const instance = MockEventSource.lastInstance;

        await act(async () => {
            instance.onmessage({
                data: JSON.stringify({ type: 'answer', content: 'The answer is 42.' }),
            });
        });

        expect(screen.getByText('The answer is 42.')).toBeDefined();
        expect(instance.close).toHaveBeenCalled();
    });

    it('renders an error event and closes the stream', async () => {
        render(<AgentChat query="test" />);
        const instance = MockEventSource.lastInstance;

        await act(async () => {
            instance.onmessage({
                data: JSON.stringify({ type: 'error', content: 'Something went wrong.' }),
            });
        });

        expect(screen.getByText('Something went wrong.')).toBeDefined();
        expect(instance.close).toHaveBeenCalled();
    });

    it('closes EventSource on onerror', async () => {
        render(<AgentChat query="test" />);
        const instance = MockEventSource.lastInstance;

        await act(async () => {
            instance.onerror(new Error('Network error'));
        });

        expect(instance.close).toHaveBeenCalled();
    });

    it('renders multiple streamed events in order', async () => {
        render(<AgentChat query="test" />);
        const instance = MockEventSource.lastInstance;

        const events = [
            { type: 'thought', content: 'First thought' },
            { type: 'action', content: 'Executing search...' },
            { type: 'observation', content: 'Found 3 results' },
        ];

        await act(async () => {
            for (const ev of events) {
                instance.onmessage({ data: JSON.stringify(ev) });
            }
        });

        expect(screen.getByText('First thought')).toBeDefined();
        expect(screen.getByText('Executing search...')).toBeDefined();
        expect(screen.getByText('Found 3 results')).toBeDefined();
    });

    it('silently ignores malformed JSON messages', async () => {
        render(<AgentChat query="test" />);
        const instance = MockEventSource.lastInstance;

        await act(async () => {
            instance.onmessage({ data: 'not-json{{' });
        });

        expect(instance.close).not.toHaveBeenCalled();
    });

    it('closes EventSource on unmount', () => {
        const { unmount } = render(<AgentChat query="live query" />);
        const instance = MockEventSource.lastInstance;
        unmount();
        expect(instance.close).toHaveBeenCalled();
    });
});
