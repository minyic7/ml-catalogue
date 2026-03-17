import { Link, Navigate, useParams } from "react-router-dom";
import { CONTENT_STRUCTURE } from "../config/content";

export default function PageView() {
  const { levelSlug, chapterSlug, pageSlug } = useParams();

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

      <h2 className="text-2xl font-bold">{page.title}</h2>
      {page.description && (
        <p className="mt-1 text-muted-foreground">{page.description}</p>
      )}
      <p className="mt-6 text-muted-foreground italic">
        Content coming soon.
      </p>
    </div>
  );
}
