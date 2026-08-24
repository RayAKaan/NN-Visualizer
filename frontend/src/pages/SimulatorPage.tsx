import React, { useState, useEffect } from "react";
import { useSessionStore } from "../store/sessionStore";
import { useSimulatorStore } from "../store/simulatorStore";
import { useComputationStore } from "../store/computationStore";
import { useArchitectureStore } from "../store/architectureStore";
import { UnifiedLayout } from "../components/layout/UnifiedLayout";
import { BuildTab } from "../components/tabs/BuildTab";
import { RunInspectTab } from "../components/tabs/RunInspectTab";
import { AnalyzeTab } from "../components/tabs/AnalyzeTab";
import { AdvancedTab } from "../components/tabs/AdvancedTab";
import { TabId } from "../store/sessionStore";
import axios from "axios";

const API_BASE = "http://127.0.0.1:8000/api";

export default function SimulatorPage() {
  const activeTab = useSessionStore((s) => s.activeTab);
  const setActiveTab = useSessionStore((s) => s.setActiveTab);
  const setDeviceInfo = useSessionStore((s) => s.setDeviceInfo);
  const setModelBuilt = useSessionStore((s) => s.setModelBuilt);
  const setDataset = useSessionStore((s) => s.setDataset);
  const setExecutionStatus = useSessionStore((s) => s.setExecutionStatus);
  const setGraphData = useSessionStore((s) => s.setGraphData);
  const setArchitecture = useSessionStore((s) => s.setArchitecture);
  
  const [initialized, setInitialized] = useState(false);

  // Quick Mode / Demo on load
  useEffect(() => {
    const initWithDemo = async () => {
      try {
        // Get device info
        const deviceRes = await axios.get(`${API_BASE}/device/info`);
        if (deviceRes.data?.data?.type) {
          setDeviceInfo(deviceRes.data.data);
        }

        // Build demo model automatically
        const demoArchitecture = [
          { type: "input", neurons: 16 },
          { type: "dense", neurons: 8, activation: "relu" },
          { type: "output", neurons: 2, activation: "softmax" }
        ];

        setArchitecture(demoArchitecture);
        // Keep the architecture builder store in sync so the network
        // visualization and BUILD tab match the demo graph.
        useArchitectureStore.getState().setLayers(demoArchitecture as any);
        
        // Build the model
        const buildRes = await axios.post(`${API_BASE}/simulator/architecture/build`, { layers: demoArchitecture });
        const buildData = buildRes.data || {};
        const graphId = buildData.graph_id;
        
        if (graphId) {
          setModelBuilt(true, graphId);
          setGraphData(buildData);

          // Register the graph in the simulator/computation stores so the
          // command-strip Forward button, "New Sample", and backward pass work.
          const simStore = useSimulatorStore.getState();
          simStore.setGraphId(graphId);

          // Run forward pass immediately
          const randomInput = Array(16).fill(0).map(() => Math.random() * 2 - 1);
          const forwardRes = await axios.post(`${API_BASE}/simulator/forward/full`, { graph_id: graphId, input: randomInput });
          const forwardData = forwardRes.data || {};

          if (forwardData.steps) {
            const outputNeurons = demoArchitecture[demoArchitecture.length - 1].neurons;
            const target = Array(outputNeurons).fill(0).map((_, i) => (i === 0 ? 1 : 0));
            simStore.setCurrentInput(randomInput);
            simStore.setCurrentTarget(target);
            simStore.setForwardMeta({
              forwardPassState: "complete",
              currentStepIndex: 0,
              totalSteps: forwardData.total_steps ?? forwardData.steps.length,
            });

            const computationStore = useComputationStore.getState();
            computationStore.setSteps(forwardData.steps);
            computationStore.setLayerOutputs(forwardData.layer_outputs || {});

            setExecutionStatus("complete");
          }
        }
        
        setDataset({
          name: "Demo Dataset",
          train_samples: 100,
          test_samples: 20,
          input_shape: [16],
          output_shape: [2]
        });
        
      } catch (error) {
        console.log("Demo initialization - backend may not be running");
        // Set default architecture even if backend fails
        setArchitecture([
          { type: "input", neurons: 16 },
          { type: "dense", neurons: 8, activation: "relu" },
          { type: "output", neurons: 2, activation: "softmax" }
        ]);
      }
      
      setInitialized(true);
    };

    initWithDemo();
  }, []);

  const renderTabContent = () => {
    switch (activeTab) {
      case "build":
        return <BuildTab />;
      case "run":
      case "inspect":
        return <RunInspectTab />;
      case "analyze":
        return <AnalyzeTab />;
      case "advanced":
        return <AdvancedTab />;
      default:
        return <BuildTab />;
    }
  };

  // Show loading while demo initializes
  if (!initialized) {
    return (
      <div className="loading-screen">
        <div className="loading-content">
          <div className="loading-spinner" />
          <h2>Neurofluxion</h2>
          <p>Loading demo model...</p>
        </div>
      </div>
    );
  }

  return <UnifiedLayout>{renderTabContent()}</UnifiedLayout>;
}