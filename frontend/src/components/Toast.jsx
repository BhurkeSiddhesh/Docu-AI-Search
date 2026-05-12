import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';

const ToastContext = createContext(null);

let nextId = 1;

export function ToastProvider({ children }) {
    const [toasts, setToasts] = useState([]);

    const dismiss = useCallback((id) => {
        setToasts((t) => t.filter((x) => x.id !== id));
    }, []);

    const push = useCallback((message, type = 'info', duration = 3500) => {
        const id = nextId++;
        setToasts((t) => [...t, { id, message, type }]);
        if (duration > 0) {
            setTimeout(() => dismiss(id), duration);
        }
        return id;
    }, [dismiss]);

    const toast = {
        success: (m) => push(m, 'success'),
        error: (m) => push(m, 'error'),
        info: (m) => push(m, 'info'),
    };

    return (
        <ToastContext.Provider value={toast}>
            {children}
            <div className="fixed bottom-4 right-4 z-[200] flex flex-col gap-2 pointer-events-none">
                {toasts.map((t) => (
                    <ToastItem key={t.id} toast={t} onDismiss={() => dismiss(t.id)} />
                ))}
            </div>
        </ToastContext.Provider>
    );
}

function ToastItem({ toast, onDismiss }) {
    const Icon = toast.type === 'success' ? CheckCircle2 : toast.type === 'error' ? AlertCircle : Info;
    const color =
        toast.type === 'success'
            ? 'bg-green-50 dark:bg-green-950/50 text-green-900 dark:text-green-100 border-green-200 dark:border-green-900'
            : toast.type === 'error'
            ? 'bg-red-50 dark:bg-red-950/50 text-red-900 dark:text-red-100 border-red-200 dark:border-red-900'
            : 'bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 border-slate-200 dark:border-slate-800';

    return (
        <div
            className={`pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-lg border shadow-lg max-w-sm animate-slide-up ${color}`}
            role="status"
        >
            <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <span className="text-sm font-medium flex-1">{toast.message}</span>
            <button onClick={onDismiss} className="opacity-60 hover:opacity-100" aria-label="Dismiss">
                <X className="w-4 h-4" />
            </button>
        </div>
    );
}

export function useToast() {
    const ctx = useContext(ToastContext);
    if (!ctx) {
        // Fallback no-op so tests / standalone renders don't crash
        return { success: () => {}, error: () => {}, info: () => {} };
    }
    return ctx;
}

export default ToastProvider;
