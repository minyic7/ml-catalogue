export type { Page, Chapter, Level } from "./types";

import type { Level } from "./types";
import { foundational } from "./foundational";
import { coreMl } from "./core-ml";
import { advanced } from "./advanced";
import { professional } from "./professional";

export const CONTENT_STRUCTURE: Level[] = [
  foundational,
  coreMl,
  advanced,
  professional,
];
