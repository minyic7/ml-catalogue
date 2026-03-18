import { useCallback, useEffect, useRef, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { CheckCircle2Icon, CircleIcon } from "lucide-react";
import { CONTENT_STRUCTURE } from "../config/content";
import { CodeBlock } from "../components/CodeBlock";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { OutputArea, type OutputData } from "../components/OutputArea";
import { RunButton, type RunMode, type DeviceType } from "../components/RunButton";
import { Button } from "../components/ui/button";
import { useProgress } from "../hooks/useProgress";
import { executeCode } from "../api/execute";

const TIMEOUT_LIMITS: Record<RunMode, number> = {
  quick: 30,
  full: 120,
};

const WARNING_THRESHOLDS: Record<RunMode, number> = {
  quick: 20,
  full: 90,
};

export default function PageView() {
  const { levelSlug, chapterSlug, pageSlug } = useParams();
  const [isLoading, setIsLoading] = useState(false);
  const [output, setOutput] = useState<OutputData | null>(null);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [timeoutWarning, setTimeoutWarning] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const modeRef = useRef<RunMode>("quick");
  const outputRef = useRef<HTMLDivElement | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearTimer();
      abortControllerRef.current?.abort();
    };
  }, [clearTimer]);

  // Scroll output into view when it changes
  useEffect(() => {
    if (output) {
      outputRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [output]);

  const level = CONTENT_STRUCTURE.find((l) => l.slug === levelSlug);
  const chapter = level?.chapters.find((c) => c.slug === chapterSlug);
  const page = chapter?.pages.find((p) => p.slug === pageSlug);

  const handleCancel = useCallback(() => {
    abortControllerRef.current?.abort();
    clearTimer();
    setIsLoading(false);
    setElapsedSeconds(0);
    setTimeoutWarning(false);
    setOutput({ warning: "Execution cancelled." });
  }, [clearTimer]);

  const handleRun = useCallback(
    async (mode: RunMode, device: DeviceType) => {
      if (!page?.codeSnippet) return;

      // Abort any previous execution
      abortControllerRef.current?.abort();
      clearTimer();

      const controller = new AbortController();
      abortControllerRef.current = controller;
      modeRef.current = mode;

      setIsLoading(true);
      setOutput(null);
      setExecutionTime(null);
      setElapsedSeconds(0);
      setTimeoutWarning(false);

      // Start elapsed timer
      const startTime = Date.now();
      timerRef.current = setInterval(() => {
        const seconds = Math.floor((Date.now() - startTime) / 1000);
        setElapsedSeconds(seconds);

        if (seconds >= WARNING_THRESHOLDS[modeRef.current]) {
          setTimeoutWarning(true);
        }

        if (seconds >= TIMEOUT_LIMITS[modeRef.current]) {
          controller.abort();
        }
      }, 1000);

      try {
        const result = await executeCode({
          code: page.codeSnippet,
          mode,
          device,
          signal: controller.signal,
        });
        setOutput({
          stdout: result.stdout || undefined,
          charts: result.charts.length > 0 ? result.charts : undefined,
          error: result.error || undefined,
        });
        setExecutionTime(result.execution_time_ms);
      } catch (err) {
        if (controller.signal.aborted) {
          const limit = TIMEOUT_LIMITS[mode];
          const elapsed = Math.floor((Date.now() - startTime) / 1000);
          if (elapsed >= limit) {
            setOutput({
              warning: `Execution timed out after ${limit}s. Try using ⚡ Quick mode for a smaller dataset.`,
            });
          } else {
            setOutput({ warning: "Execution cancelled." });
          }
        } else {
          setOutput({
            error: err instanceof Error ? err.message : "An unexpected error occurred",
          });
        }
      } finally {
        clearTimer();
        setIsLoading(false);
        setElapsedSeconds(0);
        setTimeoutWarning(false);
      }
    },
    [page?.codeSnippet, clearTimer],
  );

  if (!level) return <Navigate to="/404" replace />;

  // If no chapter specified, redirect to first chapter's first page
  if (!chapterSlug) {
    const firstChapter = level.chapters[0];
    const firstPage = firstChapter.pages[0];
    return (
      <Navigate
        to={`/${level.slug}/${firstChapter.slug}/${firstPage.slug}`}
        replace
      />
    );
  }

  if (!chapter) return <Navigate to="/404" replace />;

  // If no page specified, redirect to first page in chapter
  if (!pageSlug) {
    const firstPage = chapter.pages[0];
    return (
      <Navigate
        to={`/${level.slug}/${chapter.slug}/${firstPage.slug}`}
        replace
      />
    );
  }

  if (!page) return <Navigate to="/404" replace />;

  const hasContent = page.markdownContent || page.codeSnippet;
  const currentSlug = `${levelSlug}/${chapterSlug}/${pageSlug}`;

  return (
    <div className="animate-in fade-in duration-200">
      <div className="mb-6 flex items-center justify-between gap-4">
        <nav className="text-sm text-muted-foreground">
          <Link to={`/${level.slug}`} className="hover:text-foreground">
            {level.title}
          </Link>
          <span className="mx-2">&gt;</span>
          <Link
            to={`/${level.slug}/${chapter.slug}`}
            className="hover:text-foreground"
          >
            {chapter.title}
          </Link>
          <span className="mx-2">&gt;</span>
          <span className="text-foreground">{page.title}</span>
        </nav>
        <MarkAsReadButton slug={currentSlug} />
      </div>

      {hasContent ? (
        <div className="space-y-6 min-h-0">
          {page.markdownContent && (
            <MarkdownRenderer content={page.markdownContent} />
          )}
          {page.codeSnippet && (
            <>
              <CodeBlock
                code={page.codeSnippet}
                language={page.codeLanguage}
              />
              <RunButton
                onRun={handleRun}
                onCancel={handleCancel}
                isLoading={isLoading}
                elapsedSeconds={elapsedSeconds}
                timeoutWarning={timeoutWarning}
                showDeviceToggle={page.isDeepLearning}
              />
              <div ref={outputRef}>
                <OutputArea output={output} />
              </div>
              {executionTime !== null && (
                <p className="text-xs text-muted-foreground">
                  Completed in {(executionTime / 1000).toFixed(1)}s
                </p>
              )}
            </>
          )}
        </div>
      ) : (
        <>
          <h2 className="text-2xl font-bold">{page.title}</h2>
          {page.description && (
            <p className="mt-1 text-muted-foreground">{page.description}</p>
          )}
          <p className="mt-6 text-muted-foreground italic">
            Content coming soon.
          </p>
        </>
      )}
    </div>
  );
}

function MarkAsReadButton({ slug }: { slug: string }) {
  const { isRead, toggleRead } = useProgress();
  const read = isRead(slug);

  return (
    <Button
      variant={read ? "secondary" : "outline"}
      size="sm"
      onClick={() => toggleRead(slug)}
      className="shrink-0 gap-1.5"
    >
      {read ? (
        <>
          <CheckCircle2Icon className="size-4 text-green-500" />
          <span>Read</span>
        </>
      ) : (
        <>
          <CircleIcon className="size-4" />
          <span>Mark as read</span>
        </>
      )}
    </Button>
  );
}
