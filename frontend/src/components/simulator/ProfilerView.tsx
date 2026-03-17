import React, { useState } from "react";
import { simulatorApi } from "../../hooks/useSimulatorApi";
import { useSimulatorStore } from "../../store/simulatorStore";
import type { ProfileResponse } from "../../types/simulator";
import { NeuralPanel } from "@/design-system/components/NeuralPanel";
import { NeuralInput } from "@/design-system/components/NeuralInput";
import { NeuralButton } from "@/design-system/components/NeuralButton";

export function ProfilerView() {
  const { graphId } = useSimulatorStore();
  const [batchText, setBatchText] = useState("1,8,16");
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const parseBatch = () =>
    batchText
      .split(",")
      .map((v) => parseInt(v.trim()))
      .filter((v) => !Number.isNaN(v) && v > 0);

  const handleProfile = async () => {
    if (!graphId) return;
    setStatus("Profiling...");
    try {
      const res = await simulatorApi.profileFull(graphId, parseBatch());
      setProfile(res);
      setStatus(null);
    } catch {
      setStatus("Failed to run profile.");
    }
  };

  return (
    <div className="sim-view">
      <NeuralPanel className="view-panel" variant="base">
        <div className="view-title">Profiler</div>
        <div className="view-actions">
          <div className="view-field">
            <label className="view-label">Batch Sizes</label>
            <NeuralInput value={batchText} onChange={(e) => setBatchText(e.target.value)} />
          </div>
          <NeuralButton variant="primary" onClick={handleProfile} disabled={!graphId}>
            Run Profile
          </NeuralButton>
        </div>
        {status && <div className="view-status">{status}</div>}
      </NeuralPanel>
      {profile && (
        <NeuralPanel className="view-panel" variant="base">
          <div className="view-subtitle">Summary</div>
          <pre className="view-pre">{JSON.stringify(profile.summary, null, 2)}</pre>
          <div className="view-subtitle">Memory</div>
          <pre className="view-pre">{JSON.stringify(profile.memory, null, 2)}</pre>
        </NeuralPanel>
      )}
    </div>
  );
}
