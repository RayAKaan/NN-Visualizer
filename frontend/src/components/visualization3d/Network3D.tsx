import React, { useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { NeuralState } from "../../types";
import NeuronLayer3D from "./NeuronLayer3D";
import EdgeBundle3D from "./EdgeBundle3D";
import LayerLabel3D from "./LayerLabel3D";

const Network3D: React.FC<{ state: NeuralState | null }> = ({ state }) => {
  const controlsRef = useRef<any>(null);

  if (!state) {
    return <div className="card">Awaiting prediction...</div>;
  }

  return (
    <div className="card network3d-wrap">
      <button
        className="btn btn-secondary"
        onClick={() => {
          if (controlsRef.current) {
            controlsRef.current.reset();
          }
        }}
      >
        Reset camera
      </button>
      <Canvas camera={{ position: [6, 4, 10], fov: 50 }} style={{ height: 520 }}>
        <color attach="background" args={["#0a0e17"]} />
        <fog attach="fog" args={["#0a0e17", 8, 28]} />
        <ambientLight intensity={0.3} />
        <pointLight position={[8, 10, 8]} intensity={1.1} />
        <gridHelper args={[20, 20, "#1f2847", "#111827"]} position={[0, -4, 0]} />

        <NeuronLayer3D z={-3} activations={state.layers.hidden1} />
        <NeuronLayer3D z={0} activations={state.layers.hidden2} />
        <NeuronLayer3D z={3} activations={state.layers.output} />
        <EdgeBundle3D zFrom={-3} zTo={0} edges={state.edges.hidden1_hidden2} />
        <EdgeBundle3D zFrom={0} zTo={3} edges={state.edges.hidden2_output} />

        <LayerLabel3D position={[0, 3.6, -3]} label="Hidden 1 (128 neurons)" />
        <LayerLabel3D position={[0, 3.6, 0]} label="Hidden 2 (64 neurons)" />
        <LayerLabel3D position={[0, 3.6, 3]} label="Output (10 neurons)" />

        <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.05} minDistance={5} maxDistance={25} />
      </Canvas>
    </div>
  );
};

export default Network3D;
