import React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { NeuralState } from "../types/NeuralState";
import NeuronLayer3D from "./NeuronLayer3D";
import EdgeBundle3D from "./EdgeBundle3D";

interface Props {
  state: NeuralState;
}

const Network3D: React.FC<Props> = ({ state }) => {
  return (
    <Canvas
      camera={{ position: [0, 0, 12], fov: 50 }}
      style={{ height: 520, background: "#0b0e14", borderRadius: 18 }}
    >
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1.2} />

      <NeuronLayer3D z={-4} activations={state.layers.hidden1} />
      <NeuronLayer3D z={0} activations={state.layers.hidden2} />
      <NeuronLayer3D z={4} activations={state.layers.output} isOutput />

      <EdgeBundle3D zFrom={-4} zTo={0} edges={state.edges.hidden1_hidden2} />
      <EdgeBundle3D
        zFrom={0}
        zTo={4}
        edges={state.edges.hidden2_output}
        highlight={state.prediction}
      />

      <OrbitControls enablePan enableRotate enableZoom />
    </Canvas>
  );
};

export default Network3D;
