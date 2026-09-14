import React from 'react';
import { AlertTriangle, RotateCcw, Home } from 'lucide-react';

export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an unhandled error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReset = () => {
    try {
      localStorage.removeItem('openshorts_session_v1');
      localStorage.removeItem('openshorts_skip_landing');
    } catch (e) {
      console.warn('Failed to clear storage:', e);
    }
    window.location.hash = '#app';
    window.location.reload();
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-paper text-ink flex items-center justify-center p-6 select-text">
          <div className="max-w-md w-full card p-6 border border-rule2 bg-paper2 shadow-2xl text-center space-y-4">
            <div className="w-12 h-12 rounded-full bg-danger/10 border border-danger/30 text-danger flex items-center justify-center mx-auto">
              <AlertTriangle size={24} />
            </div>

            <div className="space-y-1">
              <h2 className="font-display lowercase text-xl text-ink">
                {this.props.fallbackTitle || 'Something interrupted the view'}
              </h2>
              <p className="text-xs text-muted leading-relaxed">
                An unexpected rendering issue occurred during video processing. Your data has not been lost.
              </p>
            </div>

            {this.state.error && (
              <div className="bg-paper p-3 rounded-input border border-rule text-left overflow-x-auto max-h-32 text-micro font-mono text-danger/90 custom-scrollbar">
                {this.state.error.toString()}
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-2.5 pt-2">
              <button
                type="button"
                onClick={this.handleReload}
                className="btn-primary flex-1 py-2 text-xs flex items-center justify-center gap-1.5"
              >
                <RotateCcw size={14} /> Reload Page
              </button>
              <button
                type="button"
                onClick={this.handleReset}
                className="btn-ghost flex-1 py-2 text-xs flex items-center justify-center gap-1.5"
              >
                <Home size={14} /> Reset Session
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
