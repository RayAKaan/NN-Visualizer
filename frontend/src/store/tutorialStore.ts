import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import type { Lesson, TutorialProgress, TutorialState, TutorialStep } from "../types/tutorial";
import { LESSONS } from "../data/tutorials";

interface TutorialStore extends TutorialState {
  openLessonSelector: () => void;
  closeLessonSelector: () => void;
  startLesson: (lessonId: string) => void;
  endLesson: () => void;
  nextStep: () => void;
  prevStep: () => void;
  jumpToStep: (index: number) => void;
  completeAction: (stepId: string) => void;
  answerQuiz: (stepId: string, answerIndex: number) => void;
  markLessonComplete: () => void;
  resetProgress: (lessonId: string) => void;
  resetAllProgress: () => void;
  getCompletedLessons: () => string[];
  getAvailableLessons: () => Lesson[];
  getCurrentStep: () => TutorialStep | null;
  getLessonProgress: (lessonId: string) => TutorialProgress | undefined;
  getOverallProgress: () => { completed: number; total: number; percentage: number };
}

export const useTutorialStore = create<TutorialStore>()(
  devtools(
    persist(
      (set, get) => ({
        isActive: false,
        currentLesson: null,
        currentStepIndex: 0,
        progress: {},
        showLessonSelector: false,
        highlightedElement: null,
        tooltipVisible: false,

        openLessonSelector() {
          set({ showLessonSelector: true });
        },

        closeLessonSelector() {
          set({ showLessonSelector: false });
        },

        startLesson(lessonId) {
          const lesson = LESSONS.find((l) => l.id === lessonId);
          if (!lesson) return;
          const existing = get().progress[lessonId];
          const startIndex = existing?.completed ? 0 : (existing?.currentStepIndex ?? 0);
          set({
            isActive: true,
            currentLesson: lesson,
            currentStepIndex: startIndex,
            showLessonSelector: false,
            tooltipVisible: true,
            highlightedElement: lesson.steps[startIndex]?.spotlightTarget ?? null,
            progress: {
              ...get().progress,
              [lessonId]: {
                lessonId,
                currentStepIndex: startIndex,
                completed: false,
                startedAt: existing?.startedAt ?? Date.now(),
                quizScores: existing?.quizScores ?? {},
              },
            },
          });
        },

        endLesson() {
          const { currentLesson, currentStepIndex, progress } = get();
          if (currentLesson) {
            set({
              progress: {
                ...progress,
                [currentLesson.id]: { ...progress[currentLesson.id], currentStepIndex },
              },
            });
          }
          set({
            isActive: false,
            currentLesson: null,
            currentStepIndex: 0,
            highlightedElement: null,
            tooltipVisible: false,
          });
        },

        nextStep() {
          const { currentLesson, currentStepIndex, progress } = get();
          if (!currentLesson) return;
          const next = currentStepIndex + 1;
          if (next >= currentLesson.steps.length) {
            get().markLessonComplete();
            return;
          }
          const step = currentLesson.steps[next];
          set({
            currentStepIndex: next,
            highlightedElement: step.spotlightTarget ?? null,
            tooltipVisible: true,
            progress: {
              ...progress,
              [currentLesson.id]: { ...progress[currentLesson.id], currentStepIndex: next },
            },
          });
        },

        prevStep() {
          const { currentLesson, currentStepIndex } = get();
          if (!currentLesson || currentStepIndex <= 0) return;
          const next = currentStepIndex - 1;
          set({
            currentStepIndex: next,
            highlightedElement: currentLesson.steps[next]?.spotlightTarget ?? null,
          });
        },

        jumpToStep(index) {
          const { currentLesson } = get();
          if (!currentLesson || index < 0 || index >= currentLesson.steps.length) return;
          set({
            currentStepIndex: index,
            highlightedElement: currentLesson.steps[index]?.spotlightTarget ?? null,
          });
        },

        completeAction(stepId) {
          const { currentLesson, currentStepIndex } = get();
          if (!currentLesson) return;
          const step = currentLesson.steps[currentStepIndex];
          if (step?.id === stepId && step.type === "action") get().nextStep();
        },

        answerQuiz(stepId, answerIndex) {
          const { currentLesson, currentStepIndex, progress } = get();
          if (!currentLesson) return;
          const step = currentLesson.steps[currentStepIndex];
          if (!step?.quiz || step.id !== stepId) return;
          const isCorrect = answerIndex === step.quiz.correctIndex;
          set({
            progress: {
              ...progress,
              [currentLesson.id]: {
                ...progress[currentLesson.id],
                quizScores: { ...progress[currentLesson.id].quizScores, [stepId]: isCorrect },
              },
            },
          });
        },

        markLessonComplete() {
          const { currentLesson, progress } = get();
          if (!currentLesson) return;
          set({
            progress: {
              ...progress,
              [currentLesson.id]: { ...progress[currentLesson.id], completed: true, completedAt: Date.now() },
            },
            isActive: false,
            currentLesson: null,
            currentStepIndex: 0,
            highlightedElement: null,
            tooltipVisible: false,
          });
        },

        resetProgress(lessonId) {
          set((state) => {
            const next = { ...state.progress };
            delete next[lessonId];
            return { progress: next };
          });
        },

        resetAllProgress() {
          set({ progress: {} });
        },

        getCompletedLessons() {
          return Object.entries(get().progress)
            .filter(([, p]) => p.completed)
            .map(([id]) => id);
        },

        getAvailableLessons() {
          const completed = get().getCompletedLessons();
          return LESSONS.filter((lesson) => lesson.prerequisites.every((p) => completed.includes(p)));
        },

        getCurrentStep() {
          const { currentLesson, currentStepIndex } = get();
          return currentLesson?.steps[currentStepIndex] ?? null;
        },

        getLessonProgress(lessonId) {
          return get().progress[lessonId];
        },

        getOverallProgress() {
          const completed = get().getCompletedLessons().length;
          const total = LESSONS.length;
          return { completed, total, percentage: total > 0 ? (completed / total) * 100 : 0 };
        },
      }),
      {
        name: "neurofluxion-tutorial-progress",
        partialize: (state) => ({ progress: state.progress }),
      },
    ),
    { name: "tutorial-store" },
  ),
);
