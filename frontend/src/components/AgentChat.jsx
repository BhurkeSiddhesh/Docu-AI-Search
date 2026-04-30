import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const AgentChat = ({ query }) => {
    const [events, setEvents] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const scrollRef = useRef(null);

    useEffect(() => {
        if (!query) return;

        setEvents([]);
        setIsLoading(true);

        const evtSource = new EventSource(`http://localhost:8000/api/agent/chat?query=${encodeURIComponent(query)}`);

        evtSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'error') {
                    setEvents(prev => [...prev, { ...data, timestamp: Date.now() }]);
                    evtSource.close();
                    setIsLoading(false);
                } else if (data.type === 'answer') {
                    setEvents(prev => [...prev, { ...data, timestamp: Date.now() }]);
                    evtSource.close();
                    setIsLoading(false);
                } else {
                    setEvents(prev => [...prev, { ...data, timestamp: Date.now() }]);
                }
            } catch (e) {
                console.error("Parse error", e);
            }
        };

        evtSource.onerror = (err) => {
            console.error("EventSource failed:", err);
            evtSource.close();
            setIsLoading(false);
        };

        return () => {
            evtSource.close();
        };
    }, [query]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [events]);

    const getIcon = (type) => {
        switch (type) {
            case 'thought': return <span className="material-symbols-outlined text-primary">psychology</span>;
            case 'action': return <span className="material-symbols-outlined text-amber-500">terminal</span>;
            case 'observation': return <span className="material-symbols-outlined text-emerald-500">database</span>;
            case 'answer': return <span className="material-symbols-outlined text-primary">verified</span>;
            case 'error': return <span className="material-symbols-outlined text-red-500">error</span>;
            default: return <span className="material-symbols-outlined">smart_toy</span>;
        }
    };

    return (
        <div className="w-full max-w-5xl mx-auto mt-12 px-4 pb-24">
            <div className="flex items-center gap-6 mb-12 px-8">
                <div className="w-16 h-16 rounded-[2rem] bg-primary/10 flex items-center justify-center relative">
                    <span className="material-symbols-outlined text-3xl text-primary" style={{ fontVariationSettings: "'FILL' 1" }}>smart_toy</span>
                    <div className="absolute inset-0 rounded-[2rem] border-2 border-primary/20 animate-pulse" />
                </div>
                <div>
                    <h2 className="text-3xl font-bold font-headline tracking-tight">Agent Reasoning</h2>
                    <p className="text-[10px] font-black opacity-40 uppercase tracking-[0.2em]">Neural ReAct Engine • Step-by-Step Logic</p>
                </div>
            </div>

            <div className="space-y-6 px-4">
                <AnimatePresence>
                    {events.map((event, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 20, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            transition={{ duration: 0.5, cubicBezier: [0.2, 0.8, 0.2, 1] }}
                            className={`rounded-[2.5rem] overflow-hidden 
                                ${event.type === 'answer' ? 'bg-white dark:bg-slate-900 shadow-2xl shadow-primary/10 border-2 border-primary/10' :
                                    event.type === 'observation' ? 'bg-[#f3f3fd] dark:bg-slate-950/40 opacity-70 mx-8' :
                                        'bg-[#f3f3fd] dark:bg-slate-900 border-l-8 border-primary/20'}`}
                        >
                            {event.type === 'answer' ? (
                                <div className="p-10 relative">
                                    <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -z-10" />
                                    
                                    <div className="flex items-center gap-4 mb-8">
                                        <div className="w-10 h-10 rounded-2xl bg-primary/10 flex items-center justify-center">
                                            <span className="material-symbols-outlined text-primary text-xl">auto_awesome</span>
                                        </div>
                                        <span className="text-xs font-black text-primary uppercase tracking-widest">Final Conclusion</span>
                                    </div>
                                    <div className="prose prose-slate dark:prose-invert max-w-none text-[#191b22] dark:text-white text-xl leading-relaxed font-medium font-body">
                                        {event.content}
                                    </div>
                                </div>
                            ) : (
                                <div className="p-8 flex gap-8 items-start">
                                    <div className="shrink-0 w-12 h-12 rounded-2xl bg-white dark:bg-slate-800 flex items-center justify-center shadow-sm">
                                        {getIcon(event.type)}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 mb-3">
                                            <span className="text-[10px] uppercase font-black tracking-widest opacity-40">{event.type}</span>
                                        </div>
                                        <div className={`text-[#191b22] dark:text-white ${event.type === 'observation' ? 'font-mono text-xs opacity-60 leading-relaxed' : 'font-bold text-base leading-relaxed'}`}>
                                            {event.content}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    ))}
                </AnimatePresence>

                {isLoading && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="flex items-center gap-6 px-12 py-8 bg-[#f3f3fd] dark:bg-slate-950/40 rounded-[2.5rem] opacity-60"
                    >
                        <div className="flex gap-2">
                            <div className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce [animation-delay:-0.3s]" />
                            <div className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce [animation-delay:-0.15s]" />
                            <div className="w-2.5 h-2.5 rounded-full bg-primary animate-bounce" />
                        </div>
                        <span className="text-[10px] font-black uppercase tracking-[0.4em] text-primary">Synthesizing intelligence</span>
                    </motion.div>
                )}

                <div ref={scrollRef} />
            </div>
        </div>
    );
};

export default AgentChat;
