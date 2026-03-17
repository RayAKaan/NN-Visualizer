import type { Lesson } from "../../types/tutorial";

export const LESSON_FORWARD_PASS: Lesson = {
  id: "forward-pass-basics",
  title: "Understanding the Forward Pass",
  description: "Learn how input transforms layer-by-layer into a prediction.",
  icon: "?",
  difficulty: "beginner",
  estimatedMinutes: 8,
  prerequisites: [],
  category: "fundamentals",
  steps: [
    {
      id: "welcome",
      type: "instruction",
      title: "Welcome to the Neural Journey",
      body: "You will follow how the network transforms your data from input to output.",
      tooltipPosition: "center",
    },
    {
      id: "select-architecture",
      type: "action",
      title: "Select ANN",
      body: "Choose ANN first to learn the basics.",
      spotlightTarget: '[data-tutorial-id="arch-selector-ann"]',
      spotlightShape: "pill",
      requiredAction: { type: "select_architecture", value: "ANN" },
    },
    {
      id: "start-pipeline",
      type: "action",
      title: "Run the Pipeline",
      body: "Press play to start stage-by-stage execution.",
      spotlightTarget: '[data-tutorial-id="play-button"]',
      spotlightShape: "circle",
      requiredAction: { type: "start_pipeline" },
    },
    {
      id: "quiz-relu",
      type: "quiz",
      title: "Quick Check",
      body: "Check your understanding of ReLU.",
      quiz: {
        question: "What does ReLU do to -0.5?",
        type: "multiple_choice",
        options: [
          { text: "Leaves it unchanged" },
          { text: "Turns it into 0" },
          { text: "Turns it into 0.5" },
        ],
        correctIndex: 1,
        explanation: "ReLU clamps negative values to zero.",
      },
    },
    {
      id: "celebration",
      type: "celebration",
      title: "Lesson Complete",
      body: "Great. You now understand the forward pass pipeline.",
      tooltipPosition: "center",
    },
  ],
};
