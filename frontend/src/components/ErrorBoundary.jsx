import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, info) {
        console.error('ErrorBoundary caught:', error, info);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div
                    role="alert"
                    className="flex flex-col items-center justify-center gap-4 p-8 rounded-2xl bg-destructive/10 border border-destructive/30 text-center"
                >
                    <p className="text-lg font-semibold text-destructive">Something went wrong</p>
                    <p className="text-sm text-muted-foreground">
                        {this.state.error?.message || 'An unexpected error occurred.'}
                    </p>
                    <button
                        type="button"
                        className="px-4 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:opacity-90 transition-opacity"
                        onClick={() => this.setState({ hasError: false, error: null })}
                    >
                        Try again
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

export default ErrorBoundary;
