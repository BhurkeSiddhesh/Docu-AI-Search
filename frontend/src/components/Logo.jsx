import React from 'react';

export function Logo({ size = 32 }) {
    return (
        <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="logo-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#007cf0" />
                    <stop offset="50%" stopColor="#00dfd8" />
                    <stop offset="100%" stopColor="#7928ca" />
                </linearGradient>
            </defs>
            <rect x="2" y="2" width="36" height="36" rx="8" fill="#171717" />
            <path d="M14 13h8a2 2 0 012 2v10a2 2 0 01-2 2h-8a2 2 0 01-2-2V15a2 2 0 012-2z" stroke="url(#logo-grad)" strokeWidth="2" fill="none" />
            <circle cx="26" cy="26" r="5" stroke="url(#logo-grad)" strokeWidth="2" fill="none" />
            <path d="M29.5 29.5L32 32" stroke="url(#logo-grad)" strokeWidth="2" strokeLinecap="round" />
        </svg>
    );
}

export default Logo;
