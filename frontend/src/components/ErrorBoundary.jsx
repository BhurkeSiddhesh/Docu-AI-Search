import React from 'react';
import { AlertTriangle } from 'lucide-react';
import logger from '../lib/logger';

export default class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, info) {
        logger.error(
            `ErrorBoundary caught: ${error.message}`,
            error.stack || info.componentStack
        );
    }

    render() {
        if (this.state.hasError) {
            return (
                <div 
                    role="alert"
                    className="min-h-screen flex items-center justify-center p-6 bg-slate-50 dark:bg-slate-950"
                >
                    <div className="card p-8 max-w-md text-center">
                        <AlertTriangle className="w-10 h-10 text-amber-500 mx-auto mb-3" />
                        <h2 className="font-semibold text-lg text-slate-900 dark:text-slate-50 mb-2">Something went wrong</h2>
                        <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                            The UI hit an unexpected error. Try reloading the page or reset the view.
                        </p>
                        <pre className="text-[11px] text-left bg-slate-100 dark:bg-slate-800 p-3 rounded-md overflow-auto max-h-32 mb-4">
                            {String(this.state.error)}
                        </pre>
                        <div className="flex gap-2">
                            <button
                                onClick={() => this.setState({ hasError: false, error: null })}
                                className="btn-secondary w-full"
                            >
                                Try again
                            </button>
                            <button
                                onClick={() => window.location.reload()}
                                className="btn-primary w-full"
                            >
                                Reload
                            </button>
                        </div>
                    </div>
                </div>
            );
        }
        return this.props.children;
    }
}
