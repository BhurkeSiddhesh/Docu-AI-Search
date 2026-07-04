import tailwindAnimate from "tailwindcss-animate";
import typography from "@tailwindcss/typography";

/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                /* ── Vercel ink / primary ── */
                primary: {
                    DEFAULT: '#171717',
                    foreground: '#ffffff',
                },
                /* ── Surfaces ── */
                canvas: {
                    DEFAULT: '#ffffff',
                    soft: '#fafafa',
                    'soft-2': '#f5f5f5',
                },
                /* ── Borders ── */
                hairline: {
                    DEFAULT: '#ebebeb',
                    strong: '#a1a1a1',
                },
                /* ── Text ── */
                ink: '#171717',
                body: '#4d4d4d',
                mute: '#888888',
                /* ── Semantic ── */
                link: {
                    DEFAULT: '#0070f3',
                    deep: '#0761d1',
                    'bg-soft': '#d3e5ff',
                },
                success: '#0070f3',
                error: {
                    DEFAULT: '#ee0000',
                    soft: '#f7d4d6',
                    deep: '#c50000',
                },
                warning: {
                    DEFAULT: '#f5a623',
                    soft: '#ffefcf',
                    deep: '#ab570a',
                },
                /* ── Brand accents ── */
                violet: {
                    DEFAULT: '#7928ca',
                    soft: '#d8ccf1',
                    deep: '#4c2889',
                },
                cyan: {
                    DEFAULT: '#50e3c2',
                    soft: '#aaffec',
                    deep: '#29bc9b',
                },
                'highlight-pink': '#ff0080',
                'highlight-magenta': '#eb367f',
                /* ── Gradients ── */
                'gradient-develop-start': '#007cf0',
                'gradient-develop-end': '#00dfd8',
                'gradient-preview-start': '#7928ca',
                'gradient-preview-end': '#ff0080',
                'gradient-ship-start': '#ff4d4d',
                'gradient-ship-end': '#f9cb28',
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'monospace'],
            },
            fontSize: {
                'display-xl': ['48px', { lineHeight: '48px', letterSpacing: '-2.4px', fontWeight: '600' }],
                'display-lg': ['32px', { lineHeight: '40px', letterSpacing: '-1.28px', fontWeight: '600' }],
                'display-md': ['24px', { lineHeight: '32px', letterSpacing: '-0.96px', fontWeight: '600' }],
                'display-sm': ['20px', { lineHeight: '28px', letterSpacing: '-0.6px', fontWeight: '600' }],
            },
            borderRadius: {
                'v-xs': '4px',
                'v-sm': '6px',
                'v-md': '8px',
                'v-lg': '12px',
                'v-xl': '16px',
                'pill-sm': '64px',
                'pill': '100px',
            },
            boxShadow: {
                'v-1': 'inset 0 0 0 1px rgba(0,0,0,0.08)',
                'v-2': '0px 1px 1px rgba(0,0,0,0.02), 0px 2px 2px rgba(0,0,0,0.04), inset 0 0 0 1px rgba(0,0,0,0.08)',
                'v-3': '0px 2px 2px rgba(0,0,0,0.04), 0px 8px 8px -8px rgba(0,0,0,0.04), inset 0 0 0 1px rgba(0,0,0,0.08)',
                'v-4': '0px 2px 2px rgba(0,0,0,0.04), 0px 8px 16px -4px rgba(0,0,0,0.04), inset 0 0 0 1px rgba(0,0,0,0.08)',
                'v-5': '0px 1px 1px rgba(0,0,0,0.02), 0px 8px 16px -4px rgba(0,0,0,0.04), 0px 24px 32px -8px rgba(0,0,0,0.06), inset 0 0 0 1px rgba(0,0,0,0.08)',
                /* Dark mode shadows */
                'v-dark-2': '0px 1px 1px rgba(0,0,0,0.3), 0px 2px 2px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(255,255,255,0.06)',
                'v-dark-3': '0px 2px 2px rgba(0,0,0,0.3), 0px 8px 8px -8px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(255,255,255,0.06)',
                'v-dark-4': '0px 2px 2px rgba(0,0,0,0.3), 0px 8px 16px -4px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(255,255,255,0.06)',
                'v-dark-5': '0px 1px 1px rgba(0,0,0,0.3), 0px 8px 16px -4px rgba(0,0,0,0.2), 0px 24px 32px -8px rgba(0,0,0,0.3), inset 0 0 0 1px rgba(255,255,255,0.06)',
            },
            keyframes: {
                "fade-in": {
                    "0%": { opacity: 0 },
                    "100%": { opacity: 1 },
                },
                "slide-up": {
                    "0%": { transform: "translateY(8px)", opacity: 0 },
                    "100%": { transform: "translateY(0)", opacity: 1 },
                },
                "gradient-x": {
                    "0%, 100%": { backgroundPosition: "0% 50%" },
                    "50%": { backgroundPosition: "100% 50%" },
                },
            },
            animation: {
                "fade-in": "fade-in 0.25s ease-out",
                "slide-up": "slide-up 0.3s ease-out",
                "gradient-x": "gradient-x 6s ease infinite",
            },
        },
    },
    plugins: [
        tailwindAnimate,
        typography,
    ],
}
