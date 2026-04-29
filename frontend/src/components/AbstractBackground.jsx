import React from 'react';
import { motion } from 'framer-motion';

const AbstractBackground = () => {
    // Soft, liquid, high-end blobs
    const blobs = [
        { id: 1, size: 900, color: 'rgba(0, 64, 224, 0.05)', duration: 30, delay: 0, x: -10, y: -10 },
        { id: 2, size: 700, color: 'rgba(124, 58, 237, 0.04)', duration: 35, delay: 2, x: 60, y: -20 },
        { id: 3, size: 800, color: 'rgba(59, 130, 246, 0.03)', duration: 40, delay: 5, x: -20, y: 50 },
        { id: 4, size: 600, color: 'rgba(139, 92, 246, 0.04)', duration: 45, delay: 0, x: 50, y: 60 },
        { id: 5, size: 400, color: 'rgba(34, 211, 238, 0.02)', duration: 25, delay: 3, x: 20, y: 20 },
    ];

    return (
        <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none bg-[#faf8ff] dark:bg-slate-950">
            {/* Mesh gradient overlay */}
            <div className="absolute inset-0 opacity-40 dark:opacity-20" style={{
                backgroundImage: `radial-gradient(at 0% 0%, rgba(0, 64, 224, 0.05) 0, transparent 50%), 
                                  radial-gradient(at 50% 0%, rgba(124, 58, 237, 0.05) 0, transparent 50%),
                                  radial-gradient(at 100% 0%, rgba(59, 130, 246, 0.05) 0, transparent 50%)`
            }} />
            
            {blobs.map((blob) => (
                <motion.div
                    key={blob.id}
                    className="absolute rounded-full blur-[160px]"
                    style={{
                        left: `${blob.x}%`,
                        top: `${blob.y}%`,
                        width: blob.size,
                        height: blob.size,
                        backgroundColor: blob.color,
                    }}
                    animate={{
                        x: [0, 80, -40, 0],
                        y: [0, -60, 30, 0],
                        scale: [1, 1.1, 0.95, 1],
                        rotate: [0, 15, -15, 0]
                    }}
                    transition={{
                        duration: blob.duration,
                        delay: blob.delay,
                        repeat: Infinity,
                        ease: "linear"
                    }}
                />
            ))}
            
            {/* Fine texture overlay */}
            <div className="absolute inset-0 opacity-[0.03] dark:opacity-[0.05] pointer-events-none mix-blend-overlay" style={{
                backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`
            }} />

            {/* Subtle glow at the bottom */}
            <div className="absolute bottom-0 left-0 right-0 h-[40vh] bg-gradient-to-t from-white dark:from-slate-950 to-transparent z-10" />
        </div>
    );
};

export default AbstractBackground;
