import type { Lesson } from "../../types/tutorial";

export const LESSON_CNN: Lesson = {
  id: "cnn-basics",
  title: "CNN for Images",
  description: "Build a CNN and explore feature maps.",
  icon: "?",
  difficulty: "intermediate",
  estimatedMinutes: 15,
  prerequisites: ["forward-pass-basics"],
  category: "cnn",
  steps: [
    {
      id: "cnn-intro",
      type: "instruction",
      title: "Load a CNN",
      body: "Choose the Simple CNN or LeNet-5 template.",
      tooltipPosition: "center",
    },
    {
      id: "load-mnist",
      type: "instruction",
      title: "Load MNIST",
      body: "In Dataset panel, pick MNIST and load samples.",
      tooltipPosition: "center",
    },
    {
      id: "view-activations",
      type: "instruction",
      title: "Activation Maps",
      body: "Open Activations tab to view feature maps.",
      tooltipPosition: "center",
    },
    {
      id: "complete",
      type: "celebration",
      title: "CNN tour complete",
      body: "You now know how CNNs build visual features.",
      tooltipPosition: "center",
    },
  ],
};
