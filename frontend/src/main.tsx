import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import './index.css';
import { ErrorBoundary } from './components/ErrorBoundary';
import RootLayout from './layouts/RootLayout';
import HomePage from './pages/HomePage';
import PageView from './pages/PageView';
import NotFound from './pages/NotFound';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route element={<RootLayout />}>
            <Route index element={<HomePage />} />
            <Route path=":levelSlug" element={<PageView />} />
            <Route path=":levelSlug/:chapterSlug" element={<PageView />} />
            <Route path=":levelSlug/:chapterSlug/:pageSlug" element={<PageView />} />
            <Route path="404" element={<NotFound />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  </StrictMode>,
);
