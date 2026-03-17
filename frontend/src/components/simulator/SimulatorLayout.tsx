import React from "react";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";

interface Props {
  header: React.ReactNode;
  left: React.ReactNode;
  center: React.ReactNode;
  right: React.ReactNode;
  footer?: React.ReactNode;
}

export function SimulatorLayout({ header, left, center, right, footer }: Props) {
  return (
    <div className="sim-shell">
      <div className="sim-header">{header}</div>
      <div className="sim-grid">
        <NeuralPanel className="sim-panel sim-left neural-scroll-area">{left}</NeuralPanel>
        <NeuralPanel className="sim-panel sim-center">{center}</NeuralPanel>
        <NeuralPanel className="sim-panel sim-right neural-scroll-area">{right}</NeuralPanel>
      </div>
      {footer ? <div className="sim-footer">{footer}</div> : null}
    </div>
  );
}
