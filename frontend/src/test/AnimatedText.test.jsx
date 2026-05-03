/**
 * AnimatedText Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import AnimatedText from '../components/AnimatedText';

describe('AnimatedText Component', () => {
    it('renders without crashing', () => {
        render(<AnimatedText text="Hello World" />);
    });

    it('renders all characters of the provided text', () => {
        const { container } = render(<AnimatedText text="Hello" />);
        const spans = container.querySelectorAll('span');
        const rendered = Array.from(spans).map(s => s.textContent).join('');
        expect(rendered).toBe('Hello');
    });

    it('renders a multi-word text with correct characters', () => {
        const { container } = render(<AnimatedText text="foo bar" />);
        const spans = container.querySelectorAll('span');
        const rendered = Array.from(spans).map(s => s.textContent).join('');
        expect(rendered).toBe('foobar');
    });

    it('applies the provided className to the container', () => {
        const { container } = render(
            <AnimatedText text="Test" className="custom-class" />
        );
        expect(container.querySelector('.custom-class')).not.toBeNull();
    });

    it('handles an empty string without crashing', () => {
        expect(() => render(<AnimatedText text="" />)).not.toThrow();
    });

    it('handles a single word', () => {
        const { container } = render(<AnimatedText text="DocuAI" />);
        const spans = container.querySelectorAll('span');
        const rendered = Array.from(spans).map(s => s.textContent).join('');
        expect(rendered).toBe('DocuAI');
    });

    it('renders one word-group div per word', () => {
        const { container } = render(<AnimatedText text="one two three" />);
        // Each word is wrapped in a plain div that holds character spans.
        // The motion container itself is the root div; its direct children are word-divs.
        const rootDiv = container.firstChild;
        const wordDivs = Array.from(rootDiv.children);
        expect(wordDivs.length).toBe(3);
    });
});
