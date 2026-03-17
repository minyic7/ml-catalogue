import { Document } from "flexsearch";
import { CONTENT_STRUCTURE } from "@/config/content";

export interface SearchResult {
  pageTitle: string;
  chapterTitle: string;
  levelTitle: string;
  slug: string;
  snippet: string;
}

interface IndexedPage {
  [key: string]: string | number;
  id: number;
  pageTitle: string;
  chapterTitle: string;
  levelTitle: string;
  slug: string;
  description: string;
  content: string;
}

function stripMarkdown(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, "") // code blocks
    .replace(/`[^`]*`/g, "") // inline code
    .replace(/!\[.*?\]\(.*?\)/g, "") // images
    .replace(/\[([^\]]*)\]\(.*?\)/g, "$1") // links
    .replace(/#{1,6}\s+/g, "") // headers
    .replace(/(\*{1,3}|_{1,3})(.*?)\1/g, "$2") // bold/italic
    .replace(/~~.*?~~/g, "") // strikethrough
    .replace(/>\s+/g, "") // blockquotes
    .replace(/[-*+]\s+/g, "") // unordered lists
    .replace(/\d+\.\s+/g, "") // ordered lists
    .replace(/\|.*\|/g, "") // tables
    .replace(/---+/g, "") // horizontal rules
    .replace(/\n{2,}/g, "\n") // multiple newlines
    .trim();
}

const pages: IndexedPage[] = [];
let index: Document<IndexedPage> | null = null;

function getIndex(): Document<IndexedPage> {
  if (index) return index;

  index = new Document<IndexedPage>({
    document: {
      id: "id",
      index: ["pageTitle", "chapterTitle", "levelTitle", "description", "content"],
      store: true,
    },
    tokenize: "forward",
    encoder: "LatinBalance",
  });

  let id = 0;
  for (const level of CONTENT_STRUCTURE) {
    for (const chapter of level.chapters) {
      for (const page of chapter.pages) {
        const doc: IndexedPage = {
          id: id++,
          pageTitle: page.title,
          chapterTitle: chapter.title,
          levelTitle: level.title,
          slug: `/${level.slug}/${chapter.slug}/${page.slug}`,
          description: page.description ?? "",
          content: stripMarkdown(page.markdownContent ?? ""),
        };
        pages.push(doc);
        index.add(doc);
      }
    }
  }

  return index;
}

function extractSnippet(content: string, query: string): string {
  const lower = content.toLowerCase();
  const queryLower = query.toLowerCase();
  const pos = lower.indexOf(queryLower);

  if (pos === -1) return content.slice(0, 120) + (content.length > 120 ? "..." : "");

  const start = Math.max(0, pos - 50);
  const end = Math.min(content.length, pos + query.length + 50);
  let snippet = "";
  if (start > 0) snippet += "...";
  snippet += content.slice(start, end);
  if (end < content.length) snippet += "...";
  return snippet;
}

export function searchPages(query: string): SearchResult[] {
  if (!query.trim()) return [];

  const idx = getIndex();
  const results = idx.search(query, { limit: 20, enrich: true });

  const seenIds = new Set<number>();
  const output: SearchResult[] = [];

  for (const fieldResult of results) {
    if (!fieldResult.result) continue;
    for (const item of fieldResult.result) {
      const docId = typeof item === "object" && item !== null ? (item as { id: number }).id : (item as number);
      if (seenIds.has(docId)) continue;
      seenIds.add(docId);

      const doc = typeof item === "object" && item !== null && "doc" in item
        ? (item as { doc: IndexedPage }).doc
        : pages[docId];

      if (!doc) continue;

      const searchText = [doc.pageTitle, doc.description, doc.content].join(" ");
      output.push({
        pageTitle: doc.pageTitle,
        chapterTitle: doc.chapterTitle,
        levelTitle: doc.levelTitle,
        slug: doc.slug,
        snippet: extractSnippet(searchText, query),
      });
    }
  }

  return output;
}
