import React from "react";
import { Line } from "@react-three/drei";
import { Edge } from "../../types";

interface Props {
  fromX: number;
  toX: number;
  fromCount: number;
  toCount: number;
  edges: Edge[];
}

export default function EdgeBundle3D({ fromX, toX, fromCount, toCount, edges }: Props) {
  const fromSize = Math.ceil(Math.sqrt(fromCount));
  const toSize = Math.ceil(Math.sqrt(toCount));

  const posFor = (idx: number, size: number) => {
    const row = Math.floor(idx / size);
    const col = idx % size;
    return [(row - size / 2) * 0.35, (col - size / 2) * 0.35] as const;
  };

  return (
    <group>
      {edges.slice(0, 160).map((e, i) => {
        const [fy, fz] = posFor(e.from, fromSize);
        const [ty, tz] = posFor(e.to, toSize);
        const abs = Math.abs(e.strength);
        const opacity = 0.1 + Math.min(abs * 3, 0.8);
        const color = e.strength >= 0 ? `rgba(6,182,212,${opacity})` : `rgba(139,92,246,${opacity})`;
        return <Line key={i} points={[[fromX, fy, fz], [toX, ty, tz]]} color={color} lineWidth={0.2 + Math.min(abs * 1.5, 1.2)} transparent opacity={opacity} />;
      })}
    </group>
  );
}
