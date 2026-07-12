import React, { createContext, useContext, useState, useCallback } from 'react';
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
    
    // Vercel-style toast colors
    const color =
        toast.type === 'success'
            ? 'bg-canvas dark:bg-[#111111] text-ink dark:text-[#ededed] border-hairline dark:border-[rgba(255,255,255,0.1)]'
            : toast.type === 'error'
            ? 'bg-error-soft dark:bg-[rgba(238,0,0,0.1)] text-error-deep dark:text-[#ffcccc] border-[rgba(238,0,0,0.2)]'
            : 'bg-canvas dark:bg-[#111111] text-ink dark:text-[#ededed] border-hairline dark:border-[rgba(255,255,255,0.1)]';

    const iconColor = 
        toast.type === 'success' ? 'text-success dark:text-[#3291ff]'
        : toast.type === 'error' ? 'text-error dark:text-[#ff9999]'
        : 'text-ink dark:text-[#ededed]';

    return (
        <div
            className={`pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-v-md border shadow-v-5 dark:shadow-v-dark-5 max-w-sm animate-slide-up ${color}`}
            role="status"
        >
            <Icon className={`w-5 h-5 flex-shrink-0 mt-0.5 ${iconColor}`} />
            <span className="text-[13px] font-medium flex-1 tracking-[-0.1px] leading-relaxed">{toast.message}</span>
            <button onClick={onDismiss} className="opacity-60 hover:opacity-100 mt-0.5" aria-label="Dismiss">
                <X className="w-4 h-4" />
            </button>
        </div>
    );
}

export function useToast() {
    const ctx = useContext(ToastContext);
    if (!ctx) {
        return { success: () => {}, error: () => {}, info: () => {} };
    }
    return ctx;
}

export default ToastProvider;
