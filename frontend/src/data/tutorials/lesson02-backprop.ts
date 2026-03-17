import type { Lesson } from "../../types/tutorial";

export const LESSON_BACKPROP: Lesson = {
  id: "backprop-basics",
  title: "Backpropagation Fundamentals",
  description: "See how gradients flow and update weights.",
  icon: "?",
  difficulty: "beginner",
  estimatedMinutes: 10,
  prerequisites: ["forward-pass-basics"],
  category: "training",
  steps: [
    {
      id: "intro",
      type: "instruction",
      title: "Why Backprop?",
      body: "Backprop computes gradients so we can update weights.",
      tooltipPosition: "center",
    },
    {
      id: "run-backward",
      type: "instruction",
      title: "Run Backward Pass",
      body: "Use the Backward panel to compute gradients for one sample.",
      tooltipPosition: "center",
    },
    {
      id: "inspect-gradients",
      type: "instruction",
      title: "Inspect Gradients",
      body: "Open the Inspector to view gradient magnitudes.",
      tooltipPosition: "center",
    },
    {
      id: "complete",
      type: "celebration",
      title: "Nice work",
      body: "You now understand how gradients flow through the network.",
      tooltipPosition: "center",
    },
  ],
};
