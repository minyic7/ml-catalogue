import type { Level } from "../types";
import { supervisedLearning } from "./supervised-learning";
import { unsupervisedLearning } from "./unsupervised-learning";
import { modelEvaluation } from "./model-evaluation";

export const coreMl: Level = {
  title: "Core ML",
  slug: "core-ml",
  icon: "cpu",
  chapters: [supervisedLearning, unsupervisedLearning, modelEvaluation],
};
