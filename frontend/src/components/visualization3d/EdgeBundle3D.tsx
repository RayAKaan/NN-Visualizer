import React, { useMemo } from "react";
import { Line } from "@react-three/drei";
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

export default function EdgeBundle3D({ sourceZ, targetZ, edges, sourceCount, targetCount, sourceColumns, targetColumns }: Props) {
  const spacing = 0.5;
  const getPos = (idx: number, count: number, cols: number, z: number): [number, number, number] => {
    const rows = Math.ceil(count / cols);
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    return [(col - cols / 2 + 0.5) * spacing, -(row - rows / 2 + 0.5) * spacing, z];
  };

  const topEdges = useMemo(() => [...edges].sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength)).slice(0, 80), [edges]);

  return (
    <group>
      {topEdges.map((edge, i) => {
        const from = getPos(edge.from, sourceCount, sourceColumns, sourceZ);
        const to = getPos(edge.to, targetCount, targetColumns, targetZ);
        const abs = Math.abs(edge.strength);
        const opacity = 0.1 + Math.min(abs * 2, 0.7);
        const color = edge.strength > 0 ? "#06b6d4" : "#8b5cf6";
        const lw = 0.5 + Math.min(abs * 3, 2.5);
        return <Line key={i} points={[from, to]} color={color} lineWidth={lw} transparent opacity={opacity} />;
      })}
    </group>
  );
}
