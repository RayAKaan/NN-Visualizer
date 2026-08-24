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

export function AdvancedTab() {
  const userMode = useSessionStore((s) => s.userMode);
  
  // Only show in Research mode, collapsed hint in Standard mode
  const isVisible = userMode === 'research';
  const isCollapsed = userMode === 'standard';
  
  return (
    <div className="tab-content advanced-tab">
      {isCollapsed ? (
        <NeuralPanel className="advanced-collapsed" variant="base">
          <div className="collapsed-message">
            <NeuralBadge tone="info">Research Mode Only</NeuralBadge>
            <p>Switch to Research mode to access advanced tools:</p>
            <ul>
              <li>Interpretability (Grad-CAM, Saliency)</li>
              <li>Adversarial Testing</li>
              <li>Model Compression</li>
              <li>Embedding Visualization</li>
              <li>Generative Models</li>
            </ul>
            <p className="hint">Set mode to "Research" in the header to unlock.</p>
          </div>
        </NeuralPanel>
      ) : isVisible ? (
        <div className="advanced-main">
          {/* Interpretability */}
          <NeuralPanel className="advanced-section" variant="base">
            <h3 className="section-title"><Eye size={14} />Interpretability</h3>
            <div className="advanced-content">
              <InterpretView />
            </div>
          </NeuralPanel>

          {/* Adversarial */}
          <NeuralPanel className="advanced-section" variant="base">
            <h3 className="section-title"><Crosshair size={14} />Adversarial Testing</h3>
            <div className="advanced-content">
              <AdversarialView />
            </div>
          </NeuralPanel>

          {/* Compression */}
          <NeuralPanel className="advanced-section" variant="base">
            <h3 className="section-title"><Package size={14} />Model Compression</h3>
            <div className="advanced-content">
              <CompressionView />
            </div>
          </NeuralPanel>

          {/* Embeddings */}
          <NeuralPanel className="advanced-section" variant="base">
            <h3 className="section-title"><ScatterChart size={14} />Embedding Visualization</h3>
            <div className="advanced-content">
              <EmbeddingsView />
            </div>
          </NeuralPanel>

          {/* Generative */}
          <NeuralPanel className="advanced-section" variant="base">
            <h3 className="section-title"><Sparkles size={14} />Generative Models</h3>
            <div className="advanced-content">
              <GenerativeView />
            </div>
          </NeuralPanel>
        </div>
      ) : (
        <NeuralPanel className="advanced-hidden" variant="base">
          <div className="hidden-message">
            <NeuralBadge tone="neutral">Not Available</NeuralBadge>
            <p>Advanced tools are only available in Research mode.</p>
          </div>
        </NeuralPanel>
      )}
    </div>
  );
}