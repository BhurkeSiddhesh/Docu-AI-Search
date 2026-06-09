import React, { useEffect, useRef, useState } from 'react';
import { Brain, Terminal, Database, CheckCircle2, AlertCircle, Bot } from 'lucide-react';

function eventIcon(type) {
    switch (type) {
        case 'thought':     return Brain;
        case 'action':      return Terminal;
        case 'observation': return Database;
        case 'answer':      return CheckCircle2;
        case 'error':       return AlertCircle;
        default:            return Bot;
    }
}

function eventColor(type) {
    switch (type) {
        case 'thought':     return 'text-indigo-500 bg-indigo-50 dark:bg-indigo-950/40';
        case 'action':      return 'text-amber-500 bg-amber-50 dark:bg-amber-950/40';
        case 'observation': return 'text-emerald-500 bg-emerald-50 dark:bg-emerald-950/40';
        case 'answer':      return 'text-primary bg-primary/10';
        case 'error':       return 'text-red-500 bg-red-50 dark:bg-red-950/40';
        default:            return 'text-slate-500 bg-slate-100 dark:bg-slate-800';
    }
}

export default function AgentView({ query }) {
    const [events, setEvents] = useState([]);
    const [isRunning, setIsRunning] = useState(true);
    const bottomRef = useRef(null);

    useEffect(() => {
        if (!query) return;
        setEvents([]);
        setIsRunning(true);

        const src = new EventSource(`/api/agent/chat?query=${encodeURIComponent(query)}`);

        src.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                setEvents((prev) => [...prev, data]);
                if (data.type === 'answer' || data.type === 'error') {
                    src.close();
                    setIsRunning(false);
                }
            } catch {
                // Malformed frame — treat as a fatal stream error so the UI
                // always reaches a terminal state rather than spinning forever.
                src.close();
                setIsRunning(false);
                setEvents((prev) => [
                    ...prev,
                    { type: 'error', content: 'Received malformed response from server.' },
                ]);
            }
        };

        src.onerror = () => {
            src.close();
            setIsRunning(false);
        };

        return () => src.close();
    }, [query]);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [events]);

    return (
        <div className="mt-8 animate-fade-in">
            <div className="flex items-center gap-3 mb-5">
                <div className="w-9 h-9 rounded-lg bg-primary text-white flex items-center justify-center">
                    <Bot className="w-5 h-5" />
                </div>
                <div>
                    <div className="font-semibold text-slate-900 dark:text-slate-50">Research agent</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400">Reasoning over your indexed documents</div>
                </div>
            </div>

            <div className="space-y-2.5">
                {events.map((evt, i) => {
                    const Icon = eventIcon(evt.type);
                    const color = eventColor(evt.type);
                    const isAnswer = evt.type === 'answer';

                    return (
                        <div
                            key={i}
                            className={`card p-4 animate-slide-up ${isAnswer ? 'border-primary/30' : ''}`}
                        >
                            <div className="flex items-start gap-3">
                                <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 ${color}`}>
                                    <Icon className="w-3.5 h-3.5" />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
                                        {evt.type}
                                    </div>
                                    <div
                                        className={`text-sm leading-relaxed whitespace-pre-wrap ${
                                            isAnswer
                                                ? 'text-slate-900 dark:text-slate-50 font-medium'
                                                : evt.type === 'observation'
                                                ? 'text-slate-500 dark:text-slate-400 font-mono text-xs'
                                                : 'text-slate-700 dark:text-slate-300'
                                        }`}
                                    >
                                        {typeof evt.content === 'string' ? evt.content : JSON.stringify(evt.content)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}

                {isRunning && (
                    <div className="card p-4 opacity-70">
                        <div className="flex items-center gap-3">
                            <div className="flex gap-1">
                                <span className="typing-dot w-1.5 h-1.5 bg-primary rounded-full inline-block" />
                                <span className="typing-dot w-1.5 h-1.5 bg-primary rounded-full inline-block" />
                                <span className="typing-dot w-1.5 h-1.5 bg-primary rounded-full inline-block" />
                            </div>
                            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
                                Reasoning…
                            </span>
                        </div>
                    </div>
                )}

                <div ref={bottomRef} />
            </div>
        </div>
    );
}
