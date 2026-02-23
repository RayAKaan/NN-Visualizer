import React from "react";
import { Bounds, OrbitControls } from "@react-three/drei";
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
    const edges = state.edges as Edge[];
    const oneThird = Math.max(1, Math.floor(edges.length / 3));
    e12 = edges.slice(0, oneThird);
    e23 = edges.slice(oneThird, oneThird * 2);
    e3o = edges.slice(oneThird * 2);
  }

  const hasSignal = h1.length || h2.length || h3.length || out.some((n: number) => n > 0);

  return (
    <div className="network-3d-container" style={{ width: "100%", height: "100%", minHeight: 460 }}>
      <Canvas camera={{ position: [0, 2.8, 14], fov: 48 }} dpr={[1, 1.75]} style={{ width: "100%", height: "100%", background: "#0a0e17", borderRadius: 10 }}>
        <ambientLight intensity={0.38} />
        <pointLight position={[10, 8, 10]} intensity={1.05} />
        <pointLight position={[-6, -2, 6]} intensity={0.35} color="#8b5cf6" />
        <fog attach="fog" args={["#0a0e17", 12, 32]} />
        <gridHelper args={[22, 22, "#1a2035", "#1a2035"]} position={[0, -6, 0]} />

        <Bounds fit clip observe margin={1.25}>
          <group>
            <LayerLabel3D position={[0, 5, -5]} text="Hidden 1 (256)" />
            <LayerLabel3D position={[0, 3.6, -1.5]} text="Hidden 2 (128)" />
            <LayerLabel3D position={[0, 3, 2]} text="Hidden 3 (64)" />
            <LayerLabel3D position={[0, 2.1, 5.5]} text={`Output (10)${hasSignal ? ` · Pred ${pred}` : ""}`} />

            <NeuronLayer3D z={-5} activations={h1} columns={16} />
            <NeuronLayer3D z={-1.5} activations={h2} columns={16} />
            <NeuronLayer3D z={2} activations={h3} columns={8} />
            <NeuronLayer3D z={5.5} activations={out} columns={10} isOutput prediction={pred} />

            <EdgeBundle3D sourceZ={-5} targetZ={-1.5} edges={e12} sourceCount={256} targetCount={128} sourceColumns={16} targetColumns={16} />
            <EdgeBundle3D sourceZ={-1.5} targetZ={2} edges={e23} sourceCount={128} targetCount={64} sourceColumns={16} targetColumns={8} />
            <EdgeBundle3D sourceZ={2} targetZ={5.5} edges={e3o} sourceCount={64} targetCount={10} sourceColumns={8} targetColumns={10} />
          </group>
        </Bounds>

        <OrbitControls enableDamping dampingFactor={0.065} rotateSpeed={0.85} minDistance={5.5} maxDistance={30} enablePan maxPolarAngle={Math.PI * 0.85} />
      </Canvas>
    </div>
  );
}
