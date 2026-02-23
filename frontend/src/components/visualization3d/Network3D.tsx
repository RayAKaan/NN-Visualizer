import React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import { Edge } from "../../types";
import NeuronLayer3D from "./NeuronLayer3D";
import EdgeBundle3D from "./EdgeBundle3D";
import LayerLabel3D from "./LayerLabel3D";

interface Props {
  hidden1: number[];
  hidden2: number[];
  output: number[];
  prediction: number;
  edgesH1H2: Edge[];
  edgesH2Out: Edge[];
}

export default function Network3D({ hidden1, hidden2, output, prediction, edgesH1H2, edgesH2Out }: Props) {
  return (
    <div className="network-3d-container">
      <Canvas camera={{ position: [0, 1.5, 7], fov: 55 }}>
        <color attach="background" args={["#0a0e17"]} />
        <fog attach="fog" args={["#0a0e17", 8, 16]} />
        <ambientLight intensity={0.45} />
        <pointLight position={[0, 6, 6]} intensity={1.2} color="#8b5cf6" />
        <pointLight position={[6, 2, 2]} intensity={0.8} color="#06b6d4" />

        <Grid args={[16, 16]} position={[0, -2.2, 0]} cellSize={0.5} cellThickness={0.4} fadeDistance={18} />

        <NeuronLayer3D x={-3.2} activations={hidden1} count={hidden1.length} />
        <NeuronLayer3D x={0} activations={hidden2} count={hidden2.length} />
        <NeuronLayer3D x={3.2} activations={output} count={output.length} winner={prediction} />

        <EdgeBundle3D fromX={-3.2} toX={0} fromCount={hidden1.length} toCount={hidden2.length} edges={edgesH1H2} />
        <EdgeBundle3D fromX={0} toX={3.2} fromCount={hidden2.length} toCount={output.length} edges={edgesH2Out} />

        <LayerLabel3D x={-3.2} label="Hidden 1" />
        <LayerLabel3D x={0} label="Hidden 2" />
        <LayerLabel3D x={3.2} label="Output" />

        <OrbitControls enablePan enableZoom enableRotate dampingFactor={0.08} />
      </Canvas>
    </div>
  );
}
