import { useCallback, useState } from "react";
import { Link, Navigate, useParams } from "react-router-dom";
import { CONTENT_STRUCTURE } from "../config/content";
import { CodeBlock } from "../components/CodeBlock";
import { MarkdownRenderer } from "../components/MarkdownRenderer";
import { OutputArea, type OutputData } from "../components/OutputArea";
import { RunButton, type RunMode } from "../components/RunButton";

export default function PageView() {
  const { levelSlug, chapterSlug, pageSlug } = useParams();
  const [isLoading, setIsLoading] = useState(false);
  const [output, setOutput] = useState<OutputData | null>(null);

  const level = CONTENT_STRUCTURE.find((l) => l.slug === levelSlug);
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

  const chapter = level.chapters.find((c) => c.slug === chapterSlug);
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

  const page = chapter.pages.find((p) => p.slug === pageSlug);
  if (!page) return <Navigate to="/404" replace />;

  const handleRun = useCallback(
    (mode: RunMode) => {
      setIsLoading(true);
      setOutput(null);
      setTimeout(() => {
        setOutput({
          stdout:
            mode === "quick"
              ? "a + b = [5 7 9]\na * 3 = [3 6 9]\na · b = 32\n||a|| = 3.7417\nâ = [0.2673 0.5345 0.8018]"
              : "a + b = [5 7 9]\na * 3 = [3 6 9]\na · b = 32\n||a|| = 3.7417\nâ = [0.2673 0.5345 0.8018]\n\n[Full mode] All assertions passed.",
        });
        setIsLoading(false);
      }, 1000);
    },
    [],
  );

  const hasContent = page.markdownContent || page.codeSnippet;

  return (
    <div>
      <nav className="mb-6 text-sm text-muted-foreground">
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

      {hasContent ? (
        <div className="space-y-6">
          {page.markdownContent && (
            <MarkdownRenderer content={page.markdownContent} />
          )}
          {page.codeSnippet && (
            <>
              <CodeBlock
                code={page.codeSnippet}
                language={page.codeLanguage}
              />
              <RunButton onRun={handleRun} isLoading={isLoading} />
              <OutputArea output={output} />
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
