import React from "react";
import { Crosshair, Eye, Package, ScatterChart, Sparkles } from "lucide-react";
import { useSessionStore } from "../../store/sessionStore";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralBadge } from "@/design-system/components/NeuralBadge";
import { InterpretView } from "../simulator/InterpretView";
import { AdversarialView } from "../simulator/AdversarialView";
import { CompressionView } from "../simulator/CompressionView";
import { EmbeddingsView } from "../simulator/EmbeddingsView";
import { GenerativeView } from "../simulator/GenerativeView";

interface ToolDisclosureProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  children: React.ReactNode;
  open?: boolean;
}

function ToolDisclosure({ icon, title, description, children, open = false }: ToolDisclosureProps) {
  return (
    <details className="sim-advanced-disclosure" open={open}>
      <summary>
        <span className="sim-advanced-summary-icon" aria-hidden="true">{icon}</span>
        <span className="sim-advanced-summary-copy"><strong>{title}</strong><small>{description}</small></span>
        <span className="sim-advanced-summary-state" aria-hidden="true">Open</span>
      </summary>
      <div className="advanced-content">{children}</div>
    </details>
  );
}

export function AdvancedTab() {
  const userMode = useSessionStore((s) => s.userMode);
  const isVisible = userMode === "research";
  const isCollapsed = userMode === "standard";

  return (
    <div className="tab-content advanced-tab">
      <section className="simulator-tab-intro">
        <span className="simulator-tab-kicker">Advanced</span>
        <h2>Ask harder questions</h2>
        <p>These tools extend the run you built. Open one only when you need interpretability, robustness, compression, embeddings, or generation.</p>
      </section>
      {isCollapsed ? (
        <NeuralPanel className="advanced-collapsed" variant="base">
          <div className="collapsed-message">
            <NeuralBadge tone="info">Research mode</NeuralBadge>
            <p>Switch to Research mode to open the advanced tool disclosures.</p>
            <ul>
              <li>Interpretability (Grad-CAM, saliency)</li>
              <li>Adversarial testing</li>
              <li>Model compression</li>
              <li>Embedding visualization</li>
              <li>Generative models</li>
            </ul>
          </div>
        </NeuralPanel>
      ) : isVisible ? (
        <div className="advanced-main">
          <ToolDisclosure icon={<Eye size={15} />} title="Interpretability" description="Grad-CAM, saliency, and attribution views" open>
            <InterpretView />
          </ToolDisclosure>
          <ToolDisclosure icon={<Crosshair size={15} />} title="Adversarial testing" description="Probe robustness with controlled perturbations">
            <AdversarialView />
          </ToolDisclosure>
          <ToolDisclosure icon={<Package size={15} />} title="Model compression" description="Pruning, quantization, and compression experiments">
            <CompressionView />
          </ToolDisclosure>
          <ToolDisclosure icon={<ScatterChart size={15} />} title="Embedding visualization" description="Inspect learned representation geometry">
            <EmbeddingsView />
          </ToolDisclosure>
          <ToolDisclosure icon={<Sparkles size={15} />} title="Generative models" description="Explore generative sampling capabilities">
            <GenerativeView />
          </ToolDisclosure>
        </div>
      ) : (
        <NeuralPanel className="advanced-hidden" variant="base">
          <div className="hidden-message">
            <NeuralBadge tone="neutral">Research mode required</NeuralBadge>
            <p>Advanced tools remain available when you switch the Simulator experience to Research.</p>
          </div>
        </NeuralPanel>
      )}
    </div>
  );
}
