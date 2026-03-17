import type { Level } from "../types";
import { modelInterpretability } from "./model-interpretability";
import { mlops } from "./mlops";
import { largeModels } from "./large-models";
import { productionDeployment } from "./production-deployment";
import { financialMl } from "./financial-ml";

export const professional: Level = {
  title: "Professional",
  slug: "professional",
  icon: "briefcase",
  chapters: [modelInterpretability, mlops, largeModels, productionDeployment, financialMl],
};
