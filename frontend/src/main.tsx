import { StrictMode, Suspense, lazy } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import { ErrorBoundary } from './components/ErrorBoundary';
import RootLayout from './layouts/RootLayout';
import { PageSkeleton } from './components/PageSkeleton';

const HomePage = lazy(() => import('./pages/HomePage'));
const PageView = lazy(() => import('./pages/PageView'));
const NotFound = lazy(() => import('./pages/NotFound'));

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter basename="/ml">
        <Routes>
          <Route element={<RootLayout />}>
            <Route index element={<Suspense fallback={<PageSkeleton />}><HomePage /></Suspense>} />
            <Route path=":levelSlug" element={<Suspense fallback={<PageSkeleton />}><PageView /></Suspense>} />
            <Route path=":levelSlug/:chapterSlug" element={<Suspense fallback={<PageSkeleton />}><PageView /></Suspense>} />
            <Route path=":levelSlug/:chapterSlug/:pageSlug" element={<Suspense fallback={<PageSkeleton />}><PageView /></Suspense>} />
            <Route path="404" element={<Suspense fallback={<PageSkeleton />}><NotFound /></Suspense>} />
            <Route path="*" element={<Suspense fallback={<PageSkeleton />}><NotFound /></Suspense>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>,
);
