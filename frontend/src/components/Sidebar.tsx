import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { CheckCircle2Icon } from "lucide-react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { CONTENT_STRUCTURE } from "@/config/content";
import { useProgress } from "@/hooks/useProgress";
import { cn } from "@/lib/utils";

export default function Sidebar({
  onNavigate,
}: {
  onNavigate?: () => void;
}) {
  const { pathname } = useLocation();
  const { isRead, getChapterProgress } = useProgress();

  // Track which levels and chapters are open.
  // Default all levels to open so the full tree is visible on first load.
  const [openLevels, setOpenLevels] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(CONTENT_STRUCTURE.map((l) => [l.slug, true])),
  );
  const [openChapters, setOpenChapters] = useState<Record<string, boolean>>(
    {},
  );

  const toggleLevel = (slug: string) =>
    setOpenLevels((prev) => ({ ...prev, [slug]: !prev[slug] }));

  const toggleChapter = (key: string) =>
    setOpenChapters((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <nav className="flex h-full flex-col overflow-y-auto py-4 text-sm">
      {CONTENT_STRUCTURE.map((level) => (
        <Collapsible
          key={level.slug}
          open={openLevels[level.slug]}
          onOpenChange={() => toggleLevel(level.slug)}
        >
          <CollapsibleTrigger className="flex w-full cursor-pointer items-center justify-between px-4 py-2 font-semibold tracking-wide text-foreground hover:bg-accent">
            <span>{level.title}</span>
            <ChevronIndicator open={!!openLevels[level.slug]} />
          </CollapsibleTrigger>

          <CollapsibleContent>
            {level.chapters.map((chapter) => {
              const chapterKey = `${level.slug}/${chapter.slug}`;
              const isChapterOpen = openChapters[chapterKey] ?? true;
              const chapterProgress = getChapterProgress(level.slug, chapter.slug);

              return (
                <Collapsible
                  key={chapterKey}
                  open={isChapterOpen}
                  onOpenChange={() => toggleChapter(chapterKey)}
                >
                  <CollapsibleTrigger className="flex w-full cursor-pointer items-center justify-between py-1.5 pl-6 pr-4 font-medium text-muted-foreground hover:bg-accent hover:text-foreground">
                    <span className="flex items-center gap-1.5">
                      {chapterProgress.completed === chapterProgress.total && chapterProgress.total > 0 && (
                        <CheckCircle2Icon className="size-3.5 shrink-0 text-green-500" />
                      )}
                      {chapter.title}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className={cn(
                        "text-xs tabular-nums",
                        chapterProgress.completed === chapterProgress.total && chapterProgress.total > 0
                          ? "text-green-500"
                          : "text-muted-foreground/60",
                      )}>
                        {chapterProgress.completed}/{chapterProgress.total}
                      </span>
                      <ChevronIndicator open={isChapterOpen} />
                    </span>
                  </CollapsibleTrigger>

                  <CollapsibleContent>
                    {chapter.pages.map((page) => {
                      const href = `/${level.slug}/${chapter.slug}/${page.slug}`;
                      const pageSlug = `${level.slug}/${chapter.slug}/${page.slug}`;
                      const isActive = pathname === href;
                      const pageRead = isRead(pageSlug);

                      return (
                        <Link
                          key={page.slug}
                          to={href}
                          onClick={onNavigate}
                          className={cn(
                            "flex items-center gap-1.5 py-1.5 pl-10 pr-4 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                            isActive &&
                              "bg-accent font-medium text-foreground",
                          )}
                        >
                          {pageRead && (
                            <CheckCircle2Icon className="size-3 shrink-0 text-green-500" />
                          )}
                          <span>{page.title}</span>
                        </Link>
                      );
                    })}
                  </CollapsibleContent>
                </Collapsible>
              );
            })}
          </CollapsibleContent>
        </Collapsible>
      ))}
    </nav>
  );
}

function ChevronIndicator({ open }: { open: boolean }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn(
        "shrink-0 transition-transform duration-200",
        open ? "rotate-90" : "rotate-0",
      )}
    >
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}
