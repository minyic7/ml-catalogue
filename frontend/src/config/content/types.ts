export interface Page {
  title: string;
  slug: string;
  description?: string;
  markdownContent?: string;
  codeSnippet?: string;
  codeLanguage?: string;
  metadata?: Record<string, unknown>;
}

export interface Chapter {
  title: string;
  slug: string;
  pages: Page[];
  metadata?: Record<string, unknown>;
}

export interface Level {
  title: string;
  slug: string;
  icon?: string;
  chapters: Chapter[];
  metadata?: Record<string, unknown>;
}
