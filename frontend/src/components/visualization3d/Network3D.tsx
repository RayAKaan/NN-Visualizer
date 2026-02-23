import React from "react";
import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { Edge } from "../../types";
import EdgeBundle3D from "./EdgeBundle3D";
import LayerLabel3D from "./LayerLabel3D";
import NeuronLayer3D from "./NeuronLayer3D";

interface Props {
  hidden1?: number[];
  hidden2?: number[];
  hidden3?: number[];
  output?: number[];
  prediction?: number;
  edgesH1H2?: Edge[];
  edgesH2H3?: Edge[];
  edgesH3Out?: Edge[];
  state?: any;
}

export default function Network3D({ hidden1 = [], hidden2 = [], hidden3 = [], output = [], prediction = 0, edgesH1H2 = [], edgesH2H3 = [], edgesH3Out = [], state = null }: Props) {
  const h1 = hidden1.length ? hidden1 : state?.layers?.hidden1 ?? [];
  const h2 = hidden2.length ? hidden2 : state?.layers?.hidden2 ?? [];
  const h3 = hidden3.length ? hidden3 : state?.layers?.hidden3 ?? [];
  const out = output.length ? output : state?.probabilities ?? Array(10).fill(0);
  const pred = Number.isFinite(prediction) ? prediction : state?.prediction ?? 0;

  let e12 = edgesH1H2;
  let e23 = edgesH2H3;
  let e3o = edgesH3Out;
  if ((!e12.length || !e23.length || !e3o.length) && state?.edges?.length) {
    const edges = state.edges;
    e12 = edges.filter((e: Edge) => e.from < 256 && e.to < 128);
    e23 = edges.filter((e: Edge) => e.from < 128 && e.to < 64);
    e3o = edges.filter((e: Edge) => e.from < 64 && e.to < 10);
  }

  return (
    <div className="network-3d-container">
      <Canvas camera={{ position: [0, 3, 16], fov: 50 }} style={{ height: "100%", background: "#0a0e17" }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 8, 10]} intensity={1.0} />
        <pointLight position={[-5, -3, 5]} intensity={0.3} color="#8b5cf6" />
        <fog attach="fog" args={["#0a0e17", 14, 30]} />
        <gridHelper args={[20, 20, "#1a2035", "#1a2035"]} position={[0, -6, 0]} />

        <LayerLabel3D position={[0, 5, -5]} text="Hidden 1 (256)" />
        <LayerLabel3D position={[0, 3.5, -1.5]} text="Hidden 2 (128)" />
        <LayerLabel3D position={[0, 3, 2]} text="Hidden 3 (64)" />
        <LayerLabel3D position={[0, 2, 5.5]} text="Output (10)" />

        <NeuronLayer3D z={-5} activations={h1} columns={16} />
        <NeuronLayer3D z={-1.5} activations={h2} columns={16} />
        <NeuronLayer3D z={2} activations={h3} columns={8} />
        <NeuronLayer3D z={5.5} activations={out} columns={10} isOutput prediction={pred} />

        <EdgeBundle3D sourceZ={-5} targetZ={-1.5} edges={e12} sourceCount={256} targetCount={128} sourceColumns={16} targetColumns={16} />
        <EdgeBundle3D sourceZ={-1.5} targetZ={2} edges={e23} sourceCount={128} targetCount={64} sourceColumns={16} targetColumns={8} />
        <EdgeBundle3D sourceZ={2} targetZ={5.5} edges={e3o} sourceCount={64} targetCount={10} sourceColumns={8} targetColumns={10} />

        <OrbitControls enableDamping dampingFactor={0.05} minDistance={6} maxDistance={28} enablePan maxPolarAngle={Math.PI * 0.85} />
      </Canvas>
    </div>
  );
}
