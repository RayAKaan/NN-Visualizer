import React from "react";
import { NeuralTabs } from "@/design-system/components/NeuralTabs";
import { PageHeader } from "@/design-system/components/PageHeader";
import { useSessionStore, TabId } from "../../store/sessionStore";
import { TabErrorBoundary } from "./TabErrorBoundary";
import { Lightbulb } from "lucide-react";

interface Tab {
  id: TabId;
  label: string;
  icon?: string;
}

const tabs: Tab[] = [
  { id: "build", label: "BUILD" },
  { id: "run", label: "RUN" },
  { id: "analyze", label: "ANALYZE" },
  { id: "advanced", label: "ADVANCED" },
];

interface Props {
  children: React.ReactNode;
}

export function UnifiedLayout({ children }: Props) {
  const activeTab = useSessionStore((s) => s.activeTab);
  const setActiveTab = useSessionStore((s) => s.setActiveTab);
  const userMode = useSessionStore((s) => s.userMode);
  const setUserMode = useSessionStore((s) => s.setUserMode);
  
  const modelBuilt = useSessionStore((s) => s.modelBuilt);
  const datasetLoaded = useSessionStore((s) => s.datasetLoaded);
  const deviceInfo = useSessionStore((s) => s.deviceInfo);
  const executionStatus = useSessionStore((s) => s.executionStatus);
  const getNextAction = useSessionStore((s) => s.getNextAction);

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId as TabId);
  };

  return (
    <div className="unified-shell">
      {/* Header */}
      <header className="unified-header">
        <PageHeader
          title="Neurofluxion Simulator"
          subtitle="Build a network, run it forward, inspect every number."
          actions={
            <label className="flex items-center gap-2 text-xs text-ink-faint">
              Mode
              <select
                value={userMode}
                onChange={(e) => setUserMode(e.target.value as any)}
                className="mode-select"
                aria-label="Experience mode"
              >
                <option value="beginner">Beginner</option>
                <option value="standard">Standard</option>
                <option value="research">Research</option>
              </select>
            </label>
          }
        >
          <nav aria-label="Simulator sections" className="w-full flex justify-center">
            <NeuralTabs
              tabs={tabs.map(t => ({ id: t.id, label: t.label }))}
              value={activeTab}
              onChange={handleTabChange}
              className="unified-tabs"
            />
          </nav>
        </PageHeader>
      </header>

      {/* Status Bar */}
      <div className="unified-status-bar">
        <div className="status-indicators">
          <div className={`status-item ${modelBuilt ? 'ready' : 'not-ready'}`}>
            <span className="status-dot" />
            <span className="status-label">Model</span>
          </div>
          <div className={`status-item ${datasetLoaded ? 'ready' : 'not-ready'}`}>
            <span className="status-dot" />
            <span className="status-label">Dataset</span>
          </div>
          <div className="status-item">
            <span className="status-label">Device:</span>
            <span className="status-value">
              {deviceInfo.type === 'gpu' ? 'GPU' : 'CPU'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Status:</span>
            <span className={`status-value execution-${executionStatus}`}>
              {executionStatus}
            </span>
          </div>
        </div>
        
        {executionStatus === 'idle' && (
          <div className="next-action-hint">
            <Lightbulb size={13} className="inline -mt-0.5 mr-1" /> {getNextAction()}
          </div>
        )}
      </div>

      {/* Main Content Area */}
      <div className="unified-main">
        <div className="unified-content">
          <TabErrorBoundary label={activeTab}>
            {children}
          </TabErrorBoundary>
        </div>
      </div>
    </div>
  );
}