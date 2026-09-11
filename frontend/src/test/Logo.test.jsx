import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Logo from '../components/Logo';

describe('Logo Component', () => {
    it('renders the app logo with default props', () => {
        render(<Logo />);
        const img = screen.getByRole('img', { name: /docu ai logo/i });
        expect(img).toBeDefined();
        expect(img.getAttribute('src')).toBe('/logo.png');
        expect(img.getAttribute('width')).toBe('32');
        expect(img.getAttribute('height')).toBe('32');
    });

    it('supports custom size and custom className', () => {
        render(<Logo size={48} className="custom-test-class" alt="Custom Logo Alt" />);
        const img = screen.getByRole('img', { name: /custom logo alt/i });
        expect(img).toBeDefined();
        expect(img.getAttribute('src')).toBe('/logo.png');
        expect(img.getAttribute('width')).toBe('48');
        expect(img.getAttribute('height')).toBe('48');
        expect(img.className).toContain('custom-test-class');
    });
});
