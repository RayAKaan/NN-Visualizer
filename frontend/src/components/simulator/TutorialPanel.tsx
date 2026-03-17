import React from "react";
import { useTutorialStore } from "../../store/tutorialStore";
import { LESSONS } from "../../data/tutorials";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function TutorialPanel() {
  const isActive = useTutorialStore((s) => s.isActive);
  const currentLesson = useTutorialStore((s) => s.currentLesson);
  const currentStepIndex = useTutorialStore((s) => s.currentStepIndex);
  const openLessonSelector = useTutorialStore((s) => s.openLessonSelector);
  const startLesson = useTutorialStore((s) => s.startLesson);
  const nextStep = useTutorialStore((s) => s.nextStep);
  const prevStep = useTutorialStore((s) => s.prevStep);
  const endLesson = useTutorialStore((s) => s.endLesson);
  const showSelector = useTutorialStore((s) => s.showLessonSelector);

  const step = currentLesson?.steps[currentStepIndex];

  return (
    <NeuralPanel className="view-panel" variant="base">
      <div className="pass-header">
        <div className="pass-title">Tutorials</div>
        <NeuralButton variant="secondary" onClick={openLessonSelector}>
          {isActive ? "Switch" : "Start"}
        </NeuralButton>
      </div>
      {showSelector ? (
        <div className="view-stack">
          {LESSONS.map((lesson) => (
            <NeuralButton
              key={lesson.id}
              onClick={() => startLesson(lesson.id)}
              variant="secondary"
            >
              {lesson.title}
            </NeuralButton>
          ))}
        </div>
      ) : null}

      {currentLesson && step ? (
        <div className="view-stack">
          <div className="view-meta">{currentLesson.title}</div>
          <div className="view-text strong">{step.title}</div>
          <div className="view-text">{step.body}</div>
          <div className="pass-actions">
            <NeuralButton variant="secondary" onClick={prevStep}>Prev</NeuralButton>
            <NeuralButton variant="secondary" onClick={nextStep}>Next</NeuralButton>
            <NeuralButton variant="danger" onClick={endLesson}>Exit</NeuralButton>
          </div>
        </div>
      ) : null}
    </NeuralPanel>
  );
}
