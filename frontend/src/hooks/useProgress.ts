import { useCallback, useSyncExternalStore } from "react";
import { CONTENT_STRUCTURE } from "@/config/content";

const STORAGE_KEY = "ml-catalogue-progress";

type ProgressData = Record<string, boolean>;

function getSnapshot(): ProgressData {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

let cachedSnapshot = getSnapshot();
const listeners = new Set<() => void>();

function subscribe(callback: () => void) {
  listeners.add(callback);
  return () => listeners.delete(callback);
}

function emitChange() {
  cachedSnapshot = getSnapshot();
  listeners.forEach((l) => l());
}

// Listen for storage changes from other tabs
if (typeof window !== "undefined") {
  window.addEventListener("storage", (e) => {
    if (e.key === STORAGE_KEY) emitChange();
  });
}

function setProgress(slug: string, read: boolean) {
  const data = getSnapshot();
  if (read) {
    data[slug] = true;
  } else {
    delete data[slug];
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  emitChange();
}

/** Derive all page slugs from the content structure. */
function getAllPageSlugs(): string[] {
  const slugs: string[] = [];
  for (const level of CONTENT_STRUCTURE) {
    for (const chapter of level.chapters) {
      for (const page of chapter.pages) {
        slugs.push(`${level.slug}/${chapter.slug}/${page.slug}`);
      }
    }
  }
  return slugs;
}

const ALL_PAGE_SLUGS = getAllPageSlugs();

export function useProgress() {
  const data = useSyncExternalStore(subscribe, () => cachedSnapshot);

  const isRead = useCallback(
    (slug: string) => !!data[slug],
    [data],
  );

  const toggleRead = useCallback((slug: string) => {
    setProgress(slug, !getSnapshot()[slug]);
  }, []);

  const totalPages = ALL_PAGE_SLUGS.length;
  const completedPages = ALL_PAGE_SLUGS.filter((s) => !!data[s]).length;

  const getChapterProgress = useCallback(
    (levelSlug: string, chapterSlug: string) => {
      const level = CONTENT_STRUCTURE.find((l) => l.slug === levelSlug);
      const chapter = level?.chapters.find((c) => c.slug === chapterSlug);
      if (!chapter) return { completed: 0, total: 0 };
      const pageSlugs = chapter.pages.map(
        (p) => `${levelSlug}/${chapterSlug}/${p.slug}`,
      );
      return {
        completed: pageSlugs.filter((s) => !!data[s]).length,
        total: pageSlugs.length,
      };
    },
    [data],
  );

  return {
    isRead,
    toggleRead,
    totalPages,
    completedPages,
    getChapterProgress,
  };
}
