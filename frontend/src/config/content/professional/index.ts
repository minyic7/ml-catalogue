import type { Level } from "../types";
import { mlops } from "./mlops";
import { largeModels } from "./large-models";
import { productionDeployment } from "./production-deployment";

export const professional: Level = {
  title: "Professional",
  slug: "professional",
  icon: "briefcase",
  chapters: [mlops, largeModels, productionDeployment],
};
