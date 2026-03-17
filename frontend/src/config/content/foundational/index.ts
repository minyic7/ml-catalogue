import type { Level } from "../types";
import { linearAlgebra } from "./linear-algebra";
import { probabilityStatistics } from "./probability-statistics";
import { pythonEssentials } from "./python-essentials";

export const foundational: Level = {
  title: "Foundational",
  slug: "foundational",
  icon: "book",
  chapters: [linearAlgebra, probabilityStatistics, pythonEssentials],
};
