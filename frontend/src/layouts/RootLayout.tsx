import { useCallback, useEffect, useState } from 'react';
import { Link, Outlet, useOutletContext } from 'react-router-dom';
import {
  Camera,
  MenuIcon,
  MessageSquare,
  SearchIcon,
  Settings,
  TextSelect,
} from 'lucide-react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import Sidebar from '@/components/Sidebar';
import SearchDialog from '@/components/SearchDialog';
import { ThemeProvider } from '@/components/ThemeProvider';
import ThemeToggle from '@/components/ThemeToggle';
import { QAAssistant, useQATools, ChatDialog } from '@/components/qa-assistant';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import {
  Sheet,
  SheetContent,
  SheetTitle,
} from '@/components/ui/sheet';

export interface LayoutContext {
  openSearch: () => void;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useLayoutContext() {
  return useOutletContext<LayoutContext>();
}

function QAHeaderButtons() {
  const { onScreenshotClick, onHighlightClick, onAskClick, onSettingsClick, hasSelection } = useQATools();

  return (
    <TooltipProvider>
      <div className="flex items-center gap-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onScreenshotClick}
              aria-label="Capture screen region to ask about"
            >
              <Camera className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Screenshot</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className={cn(
                "relative",
                hasSelection && "ring-2 ring-primary ring-offset-1 ring-offset-background"
              )}
              onClick={onHighlightClick}
              aria-label="Highlight text and ask about it"
            >
              <TextSelect className="size-4" />
              {hasSelection && (
                <span className="absolute -top-0.5 -right-0.5 flex size-2">
                  <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-75" />
                  <span className="relative inline-flex size-2 rounded-full bg-primary" />
                </span>
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Highlight</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onAskClick}
              aria-label="Ask a question about this page"
            >
              <MessageSquare className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Ask</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onSettingsClick}
              aria-label="QA settings"
            >
              <Settings className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Settings</TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}

function InlineChatPanel() {
  const { chatOpen, onChatClose, initialContext, pageContext } = useQATools();
  if (!chatOpen) return null;
  return (
    <div className="relative shrink-0">
      <ChatDialog
        isOpen={chatOpen}
        onClose={onChatClose}
        initialContext={initialContext}
        pageContext={pageContext}
      />
    </div>
  );
}

export default function RootLayout() {
  const [searchOpen, setSearchOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      setSearchOpen((prev) => !prev);
    }
  }, []);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const closeMobileSidebar = useCallback(() => setSidebarOpen(false), []);

  return (
    <ThemeProvider>
    <QAAssistant>
    <div className="flex h-full min-h-dvh flex-col">
      <header className="flex items-center justify-between border-b px-4 py-3 lg:px-6">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon-sm"
            className="lg:hidden"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open menu"
          >
            <MenuIcon className="size-5" />
          </Button>
          <Link to="/" className="cursor-pointer">
            <h1 className="text-xl font-semibold">ML Catalogue</h1>
          </Link>
        </div>
        <div className="flex items-center gap-2">
          <QAHeaderButtons />
          <div className="mx-1 h-5 w-px bg-border" />
          <Button
            variant="outline"
            size="sm"
            className="gap-2 text-muted-foreground"
            onClick={() => setSearchOpen(true)}
          >
            <SearchIcon className="size-4" />
            <span className="hidden sm:inline">Search</span>
            <kbd className="pointer-events-none hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground sm:inline-flex">
              ⌘K
            </kbd>
          </Button>
          <ThemeToggle />
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Desktop sidebar */}
        <aside className="hidden w-64 shrink-0 overflow-y-auto border-r lg:block">
          <Sidebar />
        </aside>

        {/* Mobile sidebar */}
        <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
          <SheetContent side="left" className="w-72 p-0" showCloseButton>
            <SheetTitle className="sr-only">Navigation</SheetTitle>
            <Sidebar onNavigate={closeMobileSidebar} />
          </SheetContent>
        </Sheet>

        <main className="min-w-0 flex-1 overflow-y-scroll overscroll-y-none p-4 lg:p-6" style={{ scrollbarGutter: "stable" }}>
          <ErrorBoundary>
            <Outlet context={{ openSearch: () => setSearchOpen(true) }} />
          </ErrorBoundary>
        </main>

        <InlineChatPanel />
      </div>

      <SearchDialog open={searchOpen} onOpenChange={setSearchOpen} />
    </div>
    </QAAssistant>
    </ThemeProvider>
  );
}
