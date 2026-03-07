import React, { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Line, OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { LayerInfo } from "../../types";

interface Props {
  layers: LayerInfo[];
}

interface NodePoint3D {
  id: string;
  layerIdx: number;
  nodeIdx: number;
  position: [number, number, number];
  activation: number;
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const MAX_NODES_PER_LAYER = 20;

function Neuron({ position, activation }: { position: [number, number, number]; activation: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const a = clamp01(activation);
  const color = useMemo(() => new THREE.Color().setHSL(0.52 - a * 0.2, 0.95, 0.3 + a * 0.35), [a]);

  useFrame(() => {
    if (!ref.current) return;
    ref.current.scale.setScalar(0.9 + a * 0.55);
  });

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.24, 20, 20]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.2 + a * 1.4} />
    </mesh>
  );
}

function ConnectionCurve({
  a,
  b,
  strength,
}: {
  a: [number, number, number];
  b: [number, number, number];
  strength: number;
}) {
  const points = useMemo(() => {
    const p0 = new THREE.Vector3(a[0], a[1], a[2]);
    const p2 = new THREE.Vector3(b[0], b[1], b[2]);
    const mid = p0.clone().add(p2).multiplyScalar(0.5);
    const dir = p2.clone().sub(p0);
    const normal = new THREE.Vector3(0, 1, 0)
      .cross(dir.clone().normalize())
      .normalize();
    const bend = 0.08 + strength * 0.22;
    const p1 = mid
      .clone()
      .add(normal.multiplyScalar(bend));

    const curve = new THREE.QuadraticBezierCurve3(p0, p1, p2);
    return curve.getPoints(18);
  }, [a, b, strength]);

  return (
    <>
      <Line
        points={points}
        color="#06b6d4"
        transparent
        opacity={0.04 + strength * 0.16}
        lineWidth={2}
      />
      <Line
        points={points}
        color={strength > 0.45 ? "#67e8f9" : "#334155"}
        transparent
        opacity={0.07 + strength * 0.5}
        lineWidth={1}
      />
    </>
  );
}

export default function Network3D({ layers }: Props) {
  const fixedCountsRef = useRef<Map<number, number>>(new Map());

  const graph = useMemo(() => {
    const activeLayers = layers.filter((l) => Array.isArray(l.activations) || typeof l.activations === "number");
    const nodes: NodePoint3D[] = [];
    const labels: Array<{ x: number; name: string; type: string }> = [];

    if (activeLayers.length === 0) return { nodes, labels };

    const startX = -((activeLayers.length - 1) * 3.2) / 2;
    activeLayers.forEach((layer, layerIdx) => {
      const rawActs =
        Array.isArray(layer.activations)
          ? layer.activations.map((v) => (typeof v === "number" ? v : 0))
          : typeof layer.activations === "number"
            ? [layer.activations]
            : [];
      const proposed = Math.min(Math.max(rawActs.length, 1), MAX_NODES_PER_LAYER);
      const existing = fixedCountsRef.current.get(layerIdx);
      const count = existing == null ? proposed : Math.min(MAX_NODES_PER_LAYER, Math.max(existing, proposed));
      if (existing == null || count !== existing) {
        fixedCountsRef.current.set(layerIdx, count);
      }
      const sampled =
        count === rawActs.length || rawActs.length === 0
          ? Array.from({ length: count }, (_, i) => rawActs[i] ?? 0)
          : Array.from({ length: count }, (_, i) => {
              const src = Math.floor((i / count) * rawActs.length);
              return rawActs[src] ?? 0;
            });

      const x = startX + layerIdx * 3.2;
      labels.push({ x, name: layer.name, type: layer.type });
      sampled.forEach((act, nodeIdx) => {
        const cols = Math.ceil(Math.sqrt(count));
        const rows = Math.ceil(count / cols);
        const row = Math.floor(nodeIdx / cols);
        const col = nodeIdx % cols;
        const spacing = 0.75;
        const y = (row - (rows - 1) / 2) * spacing;
        const z = (col - (cols - 1) / 2) * spacing;
        nodes.push({
          id: `${layerIdx}-${nodeIdx}`,
          layerIdx,
          nodeIdx,
          position: [x, y, z],
          activation: clamp01(act),
        });
      });
    });

    return { nodes, labels };
  }, [layers]);

  const nodesByLayer = useMemo(() => {
    const m = new Map<number, NodePoint3D[]>();
    graph.nodes.forEach((n) => {
      const list = m.get(n.layerIdx) ?? [];
      list.push(n);
      m.set(n.layerIdx, list);
    });
    return m;
  }, [graph.nodes]);

  return (
    <div className="w-full h-full rounded-lg bg-gradient-to-br from-slate-950 via-slate-900 to-cyan-950">
      <Canvas camera={{ position: [0, 0, 12], fov: 55 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[6, 6, 8]} intensity={1.4} />
        <pointLight position={[-6, -6, 8]} intensity={0.8} color="#67e8f9" />
        <OrbitControls enablePan={false} />

        {graph.labels.map((label) => (
          <group key={`${label.x}-${label.name}`}>
            <Text position={[label.x, 4.2, 0]} fontSize={0.24} color="#a5f3fc">
              {label.name}
            </Text>
            <Text position={[label.x, 3.85, 0]} fontSize={0.16} color="#94a3b8">
              {label.type}
            </Text>
          </group>
        ))}

        {graph.nodes.map((node) => (
          <Neuron key={node.id} position={node.position} activation={node.activation} />
        ))}

        {Array.from(nodesByLayer.keys())
          .sort((a, b) => a - b)
          .map((layerIdx) => {
            const src = nodesByLayer.get(layerIdx) ?? [];
            const dst = nodesByLayer.get(layerIdx + 1) ?? [];
            if (dst.length === 0) return null;

            return (
              <group key={`edges-${layerIdx}`}>
                {src.map((a) =>
                  dst.map((b) => {
                    const strength = clamp01(a.activation * 0.6 + b.activation * 0.4);
                    if (strength < 0.03) return null;
                    return (
                      <ConnectionCurve
                        key={`${a.id}-${b.id}`}
                        a={a.position}
                        b={b.position}
                        strength={strength}
                      />
                    );
                  }),
                )}
              </group>
            );
          })}
      </Canvas>
    </div>
  );
}
