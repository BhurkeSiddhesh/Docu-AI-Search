import React from 'react';

export function Logo({ size = 32, className = '', alt = 'Docu AI Logo', ...props }) {
    return (
        <img
            src="/logo.png"
            alt={alt}
            width={size}
            height={size}
            className={`object-contain select-none shrink-0 ${className}`}
            style={{ width: size, height: size }}
            draggable={false}
            {...props}
        />
    );
}

export default Logo;

