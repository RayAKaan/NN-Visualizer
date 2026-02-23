import React, { useMemo } from "react";
import { Line } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { Edge } from "../../types";

interface Props {
  sourceZ: number;
  targetZ: number;
  edges: Edge[];
  sourceCount: number;
  targetCount: number;
  sourceColumns: number;
  targetColumns: number;
}

type RenderEdge = {
  points: [number, number, number][];
  strength: number;
  lineWidth: number;
  baseOpacity: number;
  color: string;
};

export default function EdgeBundle3D({ sourceZ, targetZ, edges, sourceCount, targetCount, sourceColumns, targetColumns }: Props) {
  const spacing = 0.5;
  const tick = React.useRef(0);
  useFrame((_, delta) => {
    tick.current += delta;
  });

  const getPos = (idx: number, count: number, cols: number, z: number): [number, number, number] => {
    const rows = Math.ceil(count / cols);
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    return [(col - cols / 2 + 0.5) * spacing, -(row - rows / 2 + 0.5) * spacing, z];
  };

  const topEdges = useMemo<RenderEdge[]>(() => {
    return [...edges]
      .sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength))
      .slice(0, 100)
      .map((edge) => {
        const from = getPos(edge.from, sourceCount, sourceColumns, sourceZ);
        const to = getPos(edge.to, targetCount, targetColumns, targetZ);
        const abs = Math.abs(edge.strength);
        const opacity = 0.08 + Math.min(abs * 2.2, 0.75);
        const color = edge.strength > 0 ? "#06b6d4" : "#8b5cf6";
        const lw = 0.35 + Math.min(abs * 2.4, 2.4);
        const mid: [number, number, number] = [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2 + 0.1, (from[2] + to[2]) / 2];
        return { points: [from, mid, to], strength: edge.strength, lineWidth: lw, baseOpacity: opacity, color };
      });
  }, [edges, sourceCount, sourceColumns, sourceZ, targetCount, targetColumns, targetZ]);

  return (
    <group>
      {topEdges.map((edge, i) => {
        const pulse = 0.82 + 0.18 * Math.sin(tick.current * 2 + i * 0.1);
        return <Line key={i} points={edge.points} color={edge.color} lineWidth={edge.lineWidth} transparent opacity={edge.baseOpacity * pulse} />;
      })}
    </group>
  );
}
