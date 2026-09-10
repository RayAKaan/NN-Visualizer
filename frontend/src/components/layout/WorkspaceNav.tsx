import React from "react";
import type { LucideIcon } from "lucide-react";
import {
  BrainCircuit,
  Command,
  Database,
  FlaskConical,
  LineChart,
  Network,
  RefreshCw,
  ScanEye,
  Wifi,
  WifiOff,
} from "lucide-react";
import { cn } from "@/utils/cn";

export type WorkspaceId = "predict" | "lab" | "simulator" | "train" | "models";

export interface WorkspaceMeta {
  id: WorkspaceId;
  label: string;
  description: string;
  icon: LucideIcon;
}

export const WORKSPACES: WorkspaceMeta[] = [
  { id: "predict", label: "Prediction", description: "Use a trained model", icon: ScanEye },
  { id: "lab", label: "Lab", description: "Understand the computation", icon: FlaskConical },
  { id: "simulator", label: "Simulator", description: "Build and execute a network", icon: Network },
  { id: "train", label: "Training", description: "Train and save a model", icon: LineChart },
  { id: "models", label: "Models", description: "Discover and manage models", icon: Database },
];

export type BackendStatus = "checking" | "online" | "offline";

interface WorkspaceNavProps {
  active: WorkspaceId;
  onNavigate: (workspace: WorkspaceId) => void;
  backendStatus: BackendStatus;
  activeModel: string | null;
  onOpenCommand: () => void;
  onReload: () => void;
}

function StatusSummary({ backendStatus, activeModel }: Pick<WorkspaceNavProps, "backendStatus" | "activeModel">) {
  const status = backendStatus;
  const statusCopy: Record<BackendStatus, string> = {
    checking: "Connecting",
    online: "Backend ready",
    offline: "Backend offline",
  };
  const StatusIcon = status === "offline" ? WifiOff : Wifi;

  return (
    <div className="workspace-status" role="status" aria-live="polite">
      <span className={cn("workspace-status-dot", `is-${status}`)} aria-hidden="true" />
      <div className="workspace-status-copy">
        <span>{statusCopy[status]}</span>
        <small>{activeModel ?? "No active model"}</small>
      </div>
      <StatusIcon size={14} aria-hidden="true" />
    </div>
  );
}

export function WorkspaceNav({
  active,
  onNavigate,
  backendStatus,
  activeModel,
  onOpenCommand,
  onReload,
}: WorkspaceNavProps) {
  return (
    <aside className="workspace-sidebar" aria-label="Primary navigation">
      <div className="workspace-sidebar-inner">
        <button
          type="button"
          className="product-mark"
          onClick={() => onNavigate("predict")}
          aria-label="Neurofluxion home — go to Prediction"
        >
          <span className="product-mark-icon" aria-hidden="true"><BrainCircuit size={20} /></span>
          <span className="product-mark-copy">
            <strong>Neurofluxion</strong>
            <small>Neural network studio</small>
          </span>
        </button>

        <div className="workspace-sidebar-label">Workspaces</div>
        <nav className="workspace-nav" aria-label="Workspaces">
          {WORKSPACES.map((workspace) => {
            const Icon = workspace.icon;
            const isActive = workspace.id === active;
            return (
              <button
                type="button"
                key={workspace.id}
                className={cn("workspace-nav-item", isActive && "is-active")}
                onClick={() => onNavigate(workspace.id)}
                aria-current={isActive ? "page" : undefined}
              >
                <Icon size={18} strokeWidth={isActive ? 2.2 : 1.8} aria-hidden="true" />
                <span className="workspace-nav-item-copy">
                  <strong>{workspace.label}</strong>
                  <small>{workspace.description}</small>
                </span>
              </button>
            );
          })}
        </nav>

        <div className="workspace-sidebar-spacer" />

        <div className="workspace-sidebar-tools" aria-label="Application utilities">
          <button type="button" className="workspace-utility" onClick={onOpenCommand}>
            <Command size={16} aria-hidden="true" />
            <span>Command palette</span>
            <kbd>⌘K</kbd>
          </button>
          <button type="button" className="workspace-utility" onClick={onReload}>
            <RefreshCw size={16} aria-hidden="true" />
            <span>Reload application</span>
          </button>
        </div>
        <StatusSummary backendStatus={backendStatus} activeModel={activeModel} />
        <div className="workspace-sidebar-footer">Neurofluxion · local workspace</div>
      </div>
    </aside>
  );
}

export function MobileWorkspaceHeader({
  active,
  backendStatus,
  onOpenCommand,
}: Pick<WorkspaceNavProps, "active" | "backendStatus" | "onOpenCommand">) {
  const workspace = WORKSPACES.find((item) => item.id === active) ?? WORKSPACES[0];
  const Icon = workspace.icon;
  return (
    <header className="workspace-mobile-header">
      <div className="workspace-mobile-brand">
        <span className="product-mark-icon" aria-hidden="true"><BrainCircuit size={18} /></span>
        <span>Neurofluxion</span>
      </div>
      <div className="workspace-mobile-current">
        <Icon size={16} aria-hidden="true" />
        <strong>{workspace.label}</strong>
        <span className={cn("workspace-status-dot", `is-${backendStatus}`)} aria-label={`Backend ${backendStatus}`} />
      </div>
      <button type="button" className="workspace-mobile-command" onClick={onOpenCommand} aria-label="Open command palette">
        <Command size={18} aria-hidden="true" />
      </button>
    </header>
  );
}

export function MobileWorkspaceNav({ active, onNavigate }: Pick<WorkspaceNavProps, "active" | "onNavigate">) {
  return (
    <nav className="workspace-mobile-nav" aria-label="Workspaces">
      {WORKSPACES.map((workspace) => {
        const Icon = workspace.icon;
        const isActive = workspace.id === active;
        return (
          <button
            type="button"
            key={workspace.id}
            className={cn("workspace-mobile-nav-item", isActive && "is-active")}
            onClick={() => onNavigate(workspace.id)}
            aria-current={isActive ? "page" : undefined}
          >
            <Icon size={18} aria-hidden="true" />
            <span>{workspace.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
