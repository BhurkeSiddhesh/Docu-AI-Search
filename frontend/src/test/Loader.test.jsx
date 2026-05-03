/**
 * Loader Component Tests
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Loader } from '../components/Loader';

describe('Loader Component', () => {
    it('renders without crashing', () => {
        render(<Loader />);
    });

    it('displays the loading label text', () => {
        render(<Loader />);
        expect(screen.getByText(/Neural Matrix Loading/i)).toBeDefined();
    });

    it('renders spinning/animated elements', () => {
        const { container } = render(<Loader />);
        const animatedElements = container.querySelectorAll('.animate-spin, .animate-pulse, .animate-bounce');
        expect(animatedElements.length).toBeGreaterThan(0);
    });

    it('renders a neurology icon', () => {
        render(<Loader />);
        expect(screen.getByText('neurology')).toBeDefined();
    });
});
