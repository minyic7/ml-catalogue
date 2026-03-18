import { useState, useMemo } from "react";
import { Link } from "react-router-dom";
import { useProgress } from "@/hooks/useProgress";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { CONTENT_STRUCTURE } from "@/config/content";
import type { Level } from "@/config/content";
import {
  BookOpen,
  Cpu,
  Brain,
  Briefcase,
  ChevronDown,
  ChevronRight,
  Code,
  FileText,
  Layers,
  ArrowRight,
} from "lucide-react";

const LEVEL_META: Record<
  string,
  { icon: React.ReactNode; description: string }
> = {
  foundational: {
    icon: <BookOpen className="size-8 text-blue-500" />,
    description:
      "Linear algebra, calculus, probability, and data essentials to build your ML foundation.",
  },
  "core-ml": {
    icon: <Cpu className="size-8 text-emerald-500" />,
    description:
      "Supervised & unsupervised learning, model evaluation, feature engineering, and boosting.",
  },
  advanced: {
    icon: <Brain className="size-8 text-purple-500" />,
    description:
      "Deep learning, NLP, computer vision, time series, RL, GNNs, and generative models.",
  },
  professional: {
    icon: <Briefcase className="size-8 text-amber-500" />,
    description:
      "Interpretability, MLOps, large models, production deployment, and financial ML.",
  },
};

function useLevelStats() {
  return useMemo(() => {
    let totalChapters = 0;
    let totalSnippets = 0;

    for (const level of CONTENT_STRUCTURE) {
      totalChapters += level.chapters.length;
      for (const chapter of level.chapters) {
        for (const page of chapter.pages) {
          if (page.codeSnippet) totalSnippets++;
        }
      }
    }

    return { totalChapters, totalSnippets };
  }, []);
}

function LevelCard({
  level,
  isRead,
  getChapterProgress,
}: {
  level: Level;
  isRead: (slug: string) => boolean;
  getChapterProgress: (
    levelSlug: string,
    chapterSlug: string,
  ) => { completed: number; total: number };
}) {
  const meta = LEVEL_META[level.slug];
  const [expanded, setExpanded] = useState(false);

  const { levelCompleted, levelTotal, firstUnreadPath } = useMemo(() => {
    let completed = 0;
    let total = 0;
    let firstUnread: string | null = null;

    for (const chapter of level.chapters) {
      for (const page of chapter.pages) {
        total++;
        const slug = `${level.slug}/${chapter.slug}/${page.slug}`;
        if (isRead(slug)) {
          completed++;
        } else if (!firstUnread) {
          firstUnread = `/${slug}`;
        }
      }
    }

    return {
      levelCompleted: completed,
      levelTotal: total,
      firstUnreadPath:
        firstUnread ??
        `/${level.slug}/${level.chapters[0].slug}/${level.chapters[0].pages[0].slug}`,
    };
  }, [level, isRead]);

  const percentage =
    levelTotal > 0 ? Math.round((levelCompleted / levelTotal) * 100) : 0;
  const hasStarted = levelCompleted > 0;

  return (
    <div className="group rounded-xl border border-border bg-card shadow-sm transition-all hover:shadow-md hover:border-border/80">
      <div className="p-6">
        <div className="flex items-start gap-4">
          <div className="mt-0.5 shrink-0">{meta?.icon}</div>
          <div className="min-w-0 flex-1">
            <h3 className="text-lg font-semibold text-foreground">
              {level.title}
            </h3>
            <p className="mt-1 text-sm text-muted-foreground">
              {meta?.description}
            </p>
            <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
              <span>{level.chapters.length} chapters</span>
              <span className="text-border">|</span>
              <span>{levelTotal} pages</span>
              <span className="text-border">|</span>
              <span>{percentage}% complete</span>
            </div>
            <div className="mt-3">
              <Progress value={levelCompleted} max={levelTotal} />
            </div>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-2">
          <Button asChild size="sm">
            <Link to={firstUnreadPath}>
              {hasStarted ? "Continue" : "Start"}
              <ArrowRight className="size-3.5" data-icon="inline-end" />
            </Link>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <ChevronDown className="size-3.5" data-icon="inline-start" />
            ) : (
              <ChevronRight className="size-3.5" data-icon="inline-start" />
            )}
            Chapters
          </Button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-border px-6 py-4">
          <ul className="space-y-2">
            {level.chapters.map((chapter) => {
              const progress = getChapterProgress(level.slug, chapter.slug);
              const chapterPercentage =
                progress.total > 0
                  ? Math.round((progress.completed / progress.total) * 100)
                  : 0;
              return (
                <li key={chapter.slug}>
                  <Link
                    to={`/${level.slug}/${chapter.slug}/${chapter.pages[0].slug}`}
                    className="flex items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors hover:bg-muted"
                  >
                    <span className="font-medium text-foreground">
                      {chapter.title}
                    </span>
                    <span className="shrink-0 text-xs text-muted-foreground">
                      {progress.completed}/{progress.total} pages &middot;{" "}
                      {chapterPercentage}%
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function HomePage() {
  const { totalPages, completedPages, isRead, getChapterProgress } =
    useProgress();
  const { totalChapters, totalSnippets } = useLevelStats();
  const percentage =
    totalPages > 0 ? Math.round((completedPages / totalPages) * 100) : 0;

  return (
    <div className="mx-auto max-w-4xl space-y-10 pb-12">
      {/* Hero */}
      <div className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight text-foreground">
          ML Catalogue
        </h1>
        <p className="text-lg text-muted-foreground">
          Your interactive guide to machine learning — from foundations to
          production
        </p>
        <div className="max-w-md space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">Overall Progress</span>
            <span className="text-muted-foreground">
              {completedPages} / {totalPages} pages completed ({percentage}%)
            </span>
          </div>
          <Progress value={completedPages} max={totalPages} />
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="rounded-lg border border-border bg-card p-4 text-center">
          <FileText className="mx-auto size-5 text-muted-foreground" />
          <div className="mt-2 text-2xl font-bold text-foreground">
            {totalPages}
          </div>
          <div className="text-xs text-muted-foreground">Pages</div>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 text-center">
          <Layers className="mx-auto size-5 text-muted-foreground" />
          <div className="mt-2 text-2xl font-bold text-foreground">
            {totalChapters}
          </div>
          <div className="text-xs text-muted-foreground">Chapters</div>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 text-center">
          <Code className="mx-auto size-5 text-muted-foreground" />
          <div className="mt-2 text-2xl font-bold text-foreground">
            {totalSnippets}
          </div>
          <div className="text-xs text-muted-foreground">Code Snippets</div>
        </div>
        <div className="rounded-lg border border-border bg-card p-4 text-center">
          <BookOpen className="mx-auto size-5 text-muted-foreground" />
          <div className="mt-2 text-2xl font-bold text-foreground">
            {CONTENT_STRUCTURE.length}
          </div>
          <div className="text-xs text-muted-foreground">Levels</div>
        </div>
      </div>

      {/* Level Cards */}
      <div className="grid gap-6 md:grid-cols-2">
        {CONTENT_STRUCTURE.map((level) => (
          <LevelCard
            key={level.slug}
            level={level}
            isRead={isRead}
            getChapterProgress={getChapterProgress}
          />
        ))}
      </div>
    </div>
  );
}
