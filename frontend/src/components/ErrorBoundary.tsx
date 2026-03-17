import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangleIcon, HomeIcon, RefreshCwIcon } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  private handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="flex flex-col items-center justify-center gap-6 py-20 text-center">
          <div className="flex size-16 items-center justify-center rounded-full bg-destructive/10">
            <AlertTriangleIcon className="size-8 text-destructive" />
          </div>
          <div className="space-y-2">
            <h2 className="text-xl font-semibold">Something went wrong</h2>
            <p className="max-w-sm text-sm text-muted-foreground">
              An unexpected error occurred. You can try again or go back to the
              home page.
            </p>
          </div>
          {this.state.error && (
            <pre className="max-w-lg overflow-auto rounded-md border bg-muted px-4 py-3 text-left text-xs text-muted-foreground">
              {this.state.error.message}
            </pre>
          )}
          <div className="flex items-center gap-3">
            <Button
              variant="default"
              size="sm"
              className="gap-2"
              onClick={this.handleReset}
            >
              <RefreshCwIcon className="size-4" />
              Try again
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="gap-2"
              onClick={this.handleGoHome}
            >
              <HomeIcon className="size-4" />
              Go home
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
