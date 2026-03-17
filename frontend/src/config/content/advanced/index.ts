import type { Level } from "../types";
import { deepLearning } from "./deep-learning";
import { nlp } from "./nlp";
import { computerVision } from "./computer-vision";
import { timeSeries } from "./time-series";
import { reinforcementLearning } from "./reinforcement-learning";

export const advanced: Level = {
  title: "Advanced",
  slug: "advanced",
  icon: "brain",
  chapters: [deepLearning, nlp, computerVision, timeSeries, reinforcementLearning],
};
