import React from "react";
import { useDebugStore } from "../../store/debugStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function DebugPanel() {
  const runDiagnostics = useDebugStore((s) => s.runDiagnostics);
  const issues = useDebugStore((s) => s.issues);

  return (
    <NeuralPanel className="diag-panel" variant="base">
      <div className="diag-header">
        <div>
          <div className="diag-title">Diagnostics</div>
          <div className="diag-subtitle">{issues.length === 0 ? "No warnings." : `${issues.length} issue(s) detected`}</div>
        </div>
        <NeuralButton variant="secondary" onClick={() => void runDiagnostics()}>
          Run
        </NeuralButton>
      </div>
      <div className="diag-list neural-scroll-area">
        {issues.length === 0 && (
          <div className="diag-card diag-ok">
            <div className="diag-code">OK</div>
            <div className="diag-message">All checks healthy. Keep training.</div>
          </div>
        )}
        {issues.map((i, idx) => {
          const severity = i.severity === "critical" ? "error" : "warning";
          return (
            <div key={idx} className={`diag-card diag-${severity}`}>
              <div className="diag-code">{i.code}</div>
              <div className="diag-message">{i.message}</div>
              {i.suggestion && <div className="diag-suggestion">Suggestion: {i.suggestion}</div>}
            </div>
          );
        })}
      </div>
    </NeuralPanel>
  );
}
