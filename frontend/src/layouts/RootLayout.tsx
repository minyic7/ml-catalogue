import { useCallback, useEffect, useState } from 'react';
import { Outlet, useOutletContext } from 'react-router-dom';
import { MenuIcon, SearchIcon } from 'lucide-react';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import Sidebar from '@/components/Sidebar';
import SearchDialog from '@/components/SearchDialog';
import { ThemeProvider } from '@/components/ThemeProvider';
import ThemeToggle from '@/components/ThemeToggle';
import { QAAssistant } from '@/components/qa-assistant/QAAssistant';
import { Button } from '@/components/ui/button';
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
    <div className="flex h-screen flex-col">
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
          <h1 className="text-xl font-semibold">ML Catalogue</h1>
        </div>
        <div className="flex items-center gap-2">
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

        <main className="flex-1 overflow-auto p-4 lg:p-6">
          <ErrorBoundary>
            <Outlet context={{ openSearch: () => setSearchOpen(true) }} />
          </ErrorBoundary>
        </main>
      </div>

      <SearchDialog open={searchOpen} onOpenChange={setSearchOpen} />

      {/* QA Assistant */}
      <QAAssistant />
    </div>
    </ThemeProvider>
  );
}
