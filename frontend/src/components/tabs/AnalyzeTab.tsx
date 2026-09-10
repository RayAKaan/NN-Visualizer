import React from "react";
import { BarChart3, GitCompare, Gauge } from "lucide-react";
import { useSessionStore } from "../../store/sessionStore";
import { useTrainingSimStore } from "../../store/trainingSimStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { LiveMetricsView } from "../simulator/LiveMetricsView";
import { ProfilerView } from "../simulator/ProfilerView";

export function AnalyzeTab() {
  const userMode = useSessionStore((s) => s.userMode);
  const metricsHistory = useTrainingSimStore((s) => s.metricsHistory);
  const hasMetrics = metricsHistory.length > 0;
  const latest = metricsHistory[metricsHistory.length - 1];

  return (
    <div className="tab-content analyze-tab">
      <div className="analyze-main">
        <section className="simulator-tab-intro">
          <span className="simulator-tab-kicker">Analyze</span>
          <h2>Read the run</h2>
          <p>Metrics stay quiet until the Simulator has something to evaluate. Run forward or train first, then use these views to compare health and cost.</p>
        </section>

        <NeuralPanel className="metrics-panel" variant="base">
          <div className="section-heading-row">
            <div><h3 className="section-title"><BarChart3 size={14} /> Training metrics</h3><p className="section-subtitle">Loss and accuracy over the active run.</p></div>
            {hasMetrics ? <span className="simulator-inline-status is-ready">{metricsHistory.length} epochs</span> : null}
          </div>
          {hasMetrics ? <LiveMetricsView /> : <div className="empty-state"><p>No training metrics yet.</p><p className="hint">Open Run, load a dataset, and start a training pass to populate this view.</p></div>}
        </NeuralPanel>

        {userMode !== "beginner" ? (
          <details className="simulator-analysis-disclosure" open={hasMetrics}>
            <summary><span><Gauge size={15} /><strong>Performance profiling</strong><small>Inspect execution cost when profiling data is available.</small></span><b aria-hidden="true">Open</b></summary>
            {hasMetrics ? <div className="simulator-analysis-content"><ProfilerView /></div> : <div className="empty-state"><p>No profiling data yet.</p><p className="hint">Execute a supported pass from Run to create profiling data.</p></div>}
          </details>
        ) : null}

        {userMode !== "beginner" ? (
          <details className="simulator-analysis-disclosure">
            <summary><span><GitCompare size={15} /><strong>Model comparison</strong><small>Keep comparison work out of the primary run view.</small></span><b aria-hidden="true">Open</b></summary>
            <div className="simulator-analysis-content">
              <div className="comparison-placeholder"><p>Add multiple model runs to compare performance.</p><p className="hint">Use the comparison entry point from the relevant architecture or model workflow when you have more than one result.</p></div>
            </div>
          </details>
        ) : null}
      </div>

      <aside className="analyze-context">
        <NeuralPanel className="stats-panel" variant="sunken">
          <h4 className="context-title">Latest run snapshot</h4>
          {latest ? <div className="quick-stats">
            <div className="stat-row"><span className="stat-label">Train loss</span><span className="stat-value">{latest.train_loss?.toFixed(4) ?? "—"}</span></div>
            <div className="stat-row"><span className="stat-label">Test loss</span><span className="stat-value">{latest.test_loss?.toFixed(4) ?? "—"}</span></div>
            <div className="stat-row"><span className="stat-label">Epochs observed</span><span className="stat-value">{metricsHistory.length}</span></div>
          </div> : <p className="hint">A compact snapshot will appear after the first training epoch.</p>}
        </NeuralPanel>
      </aside>
    </div>
  );
}
