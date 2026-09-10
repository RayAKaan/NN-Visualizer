import React from "react";
import { NeuralTabs } from "@/design-system/components/NeuralTabs";
import { PageHeader } from "@/design-system/components/PageHeader";
import { useSessionStore, TabId } from "../../store/sessionStore";
import { TabErrorBoundary } from "./TabErrorBoundary";
import { Lightbulb } from "lucide-react";

interface Tab {
  id: TabId;
  label: string;
}

const tabs: Tab[] = [
  { id: "build", label: "Build" },
  { id: "run", label: "Run" },
  { id: "analyze", label: "Analyze" },
  { id: "advanced", label: "Advanced" },
];

interface Props {
  children: React.ReactNode;
  demoMode?: boolean;
}

export function UnifiedLayout({ children, demoMode = true }: Props) {
  const activeTab = useSessionStore((s) => s.activeTab);
  const setActiveTab = useSessionStore((s) => s.setActiveTab);
  const userMode = useSessionStore((s) => s.userMode);
  const setUserMode = useSessionStore((s) => s.setUserMode);
  const modelBuilt = useSessionStore((s) => s.modelBuilt);
  const datasetLoaded = useSessionStore((s) => s.datasetLoaded);
  const deviceInfo = useSessionStore((s) => s.deviceInfo);
  const executionStatus = useSessionStore((s) => s.executionStatus);
  const getNextAction = useSessionStore((s) => s.getNextAction);

  return (
    <div className="unified-shell">
      <header className="unified-header">
        <PageHeader
          eyebrow="Build · Run · Analyze · Advanced"
          title="Simulator"
          subtitle="Build a network, run it forward, and inspect every number."
          actions={
            <label className="simulator-mode-control">
              <span>Experience</span>
              <select
                value={userMode}
                onChange={(event) => setUserMode(event.target.value as "beginner" | "standard" | "research")}
                className="mode-select"
                aria-label="Simulator experience mode"
              >
                <option value="beginner">Beginner</option>
                <option value="standard">Standard</option>
                <option value="research">Research</option>
              </select>
            </label>
          }
        >
          <nav aria-label="Simulator workflow" className="simulator-workflow-nav">
            <NeuralTabs
              tabs={tabs.map((tab) => ({ id: tab.id, label: tab.label }))}
              value={activeTab}
              onChange={(value) => setActiveTab(value as TabId)}
              className="unified-tabs"
              ariaLabel="Simulator workflow"
            />
          </nav>
          {demoMode ? (
            <div className="simulator-demo-note" role="note">
              <span className="simulator-demo-badge">Demo</span>
              <span>Neurofluxion loaded a small 16 → 8 → 2 network and ran one random forward pass. Edit it in Build or run it again.</span>
            </div>
          ) : null}
        </PageHeader>
      </header>

      <section className="unified-status-bar" aria-label="Simulator status">
        <div className="simulator-status-primary">
          <span className={`simulator-status-dot is-${executionStatus}`} aria-hidden="true" />
          <strong>{executionStatus === "idle" ? "Ready" : executionStatus === "complete" ? "Complete" : executionStatus === "running" ? "Running" : "Needs attention"}</strong>
          <span>{modelBuilt ? "Network built" : "No network built"}</span>
        </div>
        <div className="simulator-status-details">
          <span>{deviceInfo.type === "gpu" ? "GPU" : "CPU"}</span>
          <span>{datasetLoaded ? "Dataset loaded" : "Dataset not loaded"}</span>
          {executionStatus === "idle" ? <span className="simulator-next-action"><Lightbulb size={13} aria-hidden="true" /> {getNextAction().replace(/^▶️\s*/, "")}</span> : null}
        </div>
      </section>

      <div className="unified-main">
        <div
          id={`tabpanel-${activeTab}`}
          role="tabpanel"
          aria-labelledby={`tab-${activeTab}`}
          className="unified-content"
        >
          <TabErrorBoundary label={activeTab}>
            {children}
          </TabErrorBoundary>
        </div>
      </div>
    </div>
  );
}
