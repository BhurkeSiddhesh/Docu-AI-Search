import React from 'react';

export const AppLogo = ({ className = "w-12 h-12" }) => (
    <div className={`${className} relative group`}>
        <svg viewBox="0 0 100 100" className="w-full h-full drop-shadow-2xl" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#0040e0" />
                    <stop offset="100%" stopColor="#5c6bff" />
                </linearGradient>
                <filter id="logoGlow" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
            </defs>

            {/* Premium Document Shell */}
            <rect x="25" y="20" width="50" height="60" rx="16" fill="white" className="dark:fill-slate-800" stroke="#f3f3fd" strokeWidth="1" />
            <rect x="35" y="35" width="30" height="4" rx="2" fill="#f3f3fd" className="dark:fill-slate-700" />
            <rect x="35" y="45" width="30" height="4" rx="2" fill="#f3f3fd" className="dark:fill-slate-700" />
            <rect x="35" y="55" width="15" height="4" rx="2" fill="#f3f3fd" className="dark:fill-slate-700" />

            {/* Neural Connection Circle */}
            <circle cx="70" cy="70" r="18" fill="url(#logoGradient)" filter="url(#logoGlow)" className="animate-pulse" />
            <path d="M70 62 L70 78 M62 70 L78 70" stroke="white" strokeWidth="4" strokeLinecap="round" />
            
            {/* Liquid Connector */}
            <path d="M60 40 Q75 40 70 60" stroke="url(#logoGradient)" strokeWidth="3" strokeLinecap="round" fill="none" strokeDasharray="4 4" />
        </svg>
    </div>
);
