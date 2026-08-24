import React from "react";
import { useSessionStore } from "../../store/sessionStore";
import { useTrainingSimStore } from "../../store/trainingSimStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { LiveMetricsView } from "../simulator/LiveMetricsView";
import { ProfilerView } from "../simulator/ProfilerView";

export function AnalyzeTab() {
  const userMode = useSessionStore((s) => s.userMode);
  const metricsHistory = useTrainingSimStore((s) => s.metricsHistory);
  
  const hasMetrics = metricsHistory.length > 0;
  const hasProfiling = false;

  return (
    <div className="tab-content analyze-tab">
      <div className="analyze-main">
        {/* Metrics Section */}
        <NeuralPanel className="metrics-panel" variant="base">
          <h3 className="section-title">Training Metrics</h3>
          
          {hasMetrics ? (
            <LiveMetricsView />
          ) : (
            <div className="empty-state">
              <p>No training metrics yet.</p>
              <p className="hint">Run training in the RUN tab to see metrics here.</p>
            </div>
          )}
        </NeuralPanel>

        {/* Profiling Section */}
        {userMode !== 'beginner' && (
          <NeuralPanel className="profiling-panel" variant="base">
            <h3 className="section-title">Performance Profiling</h3>
            
            {hasProfiling ? (
              <ProfilerView />
            ) : (
              <div className="empty-state">
                <p>No profiling data yet.</p>
                <p className="hint">Profiling runs automatically with execution.</p>
              </div>
            )}
          </NeuralPanel>
        )}

        {/* Comparison (Standard/Research mode) */}
        {userMode !== 'beginner' && (
          <NeuralPanel className="comparison-panel" variant="base">
            <h3 className="section-title">Model Comparison</h3>
            <div className="comparison-placeholder">
              <p>Add multiple models to compare performance.</p>
              <p className="hint">Use the Compare feature to train and compare different architectures.</p>
            </div>
          </NeuralPanel>
        )}
      </div>

      {/* Context Panel - Quick Stats */}
      <div className="analyze-context">
        <NeuralPanel className="stats-panel" variant="base">
          <h4 className="context-title">Quick Stats</h4>
          
          {hasMetrics && (
            <div className="quick-stats">
              <div className="stat-row">
                <span className="stat-label">Latest Train Loss:</span>
                <span className="stat-value">
                  {metricsHistory[metricsHistory.length - 1]?.train_loss?.toFixed(4) || 'N/A'}
                </span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Latest Test Loss:</span>
                <span className="stat-value">
                  {metricsHistory[metricsHistory.length - 1]?.test_loss?.toFixed(4) || 'N/A'}
                </span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Total Epochs:</span>
                <span className="stat-value">{metricsHistory.length}</span>
              </div>
            </div>
          )}
          
          {!hasMetrics && (
            <div className="no-stats">
              <p>Run training to see statistics</p>
            </div>
          )}
        </NeuralPanel>
      </div>
    </div>
  );
}