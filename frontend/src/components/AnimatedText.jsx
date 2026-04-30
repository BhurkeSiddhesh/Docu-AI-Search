import React from 'react';
import { motion } from 'framer-motion';

const AnimatedText = ({ text, className = "" }) => {
    const words = text.split(" ");

    const container = {
        hidden: { opacity: 0 },
        visible: (i = 1) => ({
            opacity: 1,
            transition: { staggerChildren: 0.08, delayChildren: 0.1 * i },
        }),
    };

    const child = {
        visible: {
            opacity: 1,
            y: 0,
            filter: "blur(0px)",
            transition: {
                type: "spring",
                damping: 20,
                stiffness: 100,
            },
        },
        hidden: {
            opacity: 0,
            y: 20,
            filter: "blur(10px)",
        },
    };

    return (
        <motion.div
            style={{ display: "flex", flexWrap: "wrap" }}
            variants={container}
            initial="hidden"
            animate="visible"
            className={className}
        >
            {words.map((word, index) => (
                <div key={index} style={{ display: "flex", marginRight: "0.25em" }}>
                    {Array.from(word).map((character, charIndex) => (
                        <motion.span variants={child} key={charIndex}>
                            {character}
                        </motion.span>
                    ))}
                </div>
            ))}
        </motion.div>
    );
};

export default AnimatedText;
