import type { Architecture, Dataset, PassDirection } from "./pipeline";

export interface Lesson {
  id: string;
  title: string;
  description: string;
  icon: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  estimatedMinutes: number;
  prerequisites: string[];
  steps: TutorialStep[];
  category: "fundamentals" | "cnn" | "rnn" | "training" | "advanced";
}

export interface TutorialStep {
  id: string;
  type: "instruction" | "action" | "quiz" | "observation" | "celebration";
  title: string;
  body: string;
  spotlightTarget?: string;
  spotlightShape?: "rect" | "circle" | "pill";
  spotlightPadding?: number;
  tooltipPosition?: "top" | "bottom" | "left" | "right" | "center";
  scrollToTarget?: boolean;
  requiredAction?: {
    type:
      | "click"
      | "draw"
      | "select_architecture"
      | "start_pipeline"
      | "step_forward"
      | "hover"
      | "change_weight"
      | "select_preset"
      | "toggle_comparison"
      | "answer_quiz";
    target?: string;
    value?: string | number | boolean;
  };
  quiz?: QuizQuestion;
  autoAdvanceMs?: number;
  showIf?: {
    architecture?: Architecture;
    dataset?: Dataset;
    passDirection?: PassDirection;
    stageId?: string;
  };
}

export interface QuizQuestion {
  question: string;
  type: "multiple_choice" | "true_false" | "visual_pick";
  options: QuizOption[];
  correctIndex: number;
  explanation: string;
  relatedVisualization?: string;
}

export interface QuizOption {
  text: string;
  icon?: string;
}

export interface TutorialProgress {
  lessonId: string;
  currentStepIndex: number;
  completed: boolean;
  startedAt: number;
  completedAt?: number;
  quizScores: Record<string, boolean>;
}

export interface TutorialState {
  isActive: boolean;
  currentLesson: Lesson | null;
  currentStepIndex: number;
  progress: Record<string, TutorialProgress>;
  showLessonSelector: boolean;
  highlightedElement: string | null;
  tooltipVisible: boolean;
}
