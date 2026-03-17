import { useState } from "react";
import { Link, useLocation } from "react-router-dom";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { CONTENT_STRUCTURE } from "@/config/content";
import { cn } from "@/lib/utils";

export default function Sidebar({
  onNavigate,
}: {
  onNavigate?: () => void;
}) {
  const { pathname } = useLocation();

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

              return (
                <Collapsible
                  key={chapterKey}
                  open={isChapterOpen}
                  onOpenChange={() => toggleChapter(chapterKey)}
                >
                  <CollapsibleTrigger className="flex w-full cursor-pointer items-center justify-between py-1.5 pl-6 pr-4 font-medium text-muted-foreground hover:bg-accent hover:text-foreground">
                    <span>{chapter.title}</span>
                    <ChevronIndicator open={isChapterOpen} />
                  </CollapsibleTrigger>

                  <CollapsibleContent>
                    {chapter.pages.map((page) => {
                      const href = `/${level.slug}/${chapter.slug}/${page.slug}`;
                      const isActive = pathname === href;

                      return (
                        <Link
                          key={page.slug}
                          to={href}
                          onClick={onNavigate}
                          className={cn(
                            "block py-1.5 pl-10 pr-4 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
                            isActive &&
                              "bg-accent font-medium text-foreground",
                          )}
                        >
                          {page.title}
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
