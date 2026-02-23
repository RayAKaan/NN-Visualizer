import React, { useMemo } from "react";

interface Props {
  z: number;
  activations: number[];
  columns: number;
  isOutput?: boolean;
  prediction?: number;
}

function actColor(a: number): string {
  const c = Math.max(0, Math.min(1, a));
  if (c <= 0.5) {
    const t = c / 0.5;
    return `rgb(${Math.round(10 + t * 129)},${Math.round(14 + t * 78)},${Math.round(23 + t * 223)})`;
  }
  const t = (c - 0.5) / 0.5;
  return `rgb(${Math.round(139 + t * (6 - 139))},${Math.round(92 + t * 90)},${Math.round(246 + t * (212 - 246))})`;
}

export default function NeuronLayer3D({ z, activations, columns, isOutput, prediction }: Props) {
  const spacing = 0.5;
  const neurons = useMemo(() => {
    const rows = Math.ceil(activations.length / columns);
    return activations.map((a, i) => {
      const col = i % columns;
      const row = Math.floor(i / columns);
      const x = (col - columns / 2 + 0.5) * spacing;
      const y = -(row - rows / 2 + 0.5) * spacing;
      const clamped = Math.max(0, Math.min(1, a || 0));
      const scale = 0.12 + clamped * 0.33;
      const isWinner = isOutput ? i === prediction : false;
      const color = isWinner ? "#06b6d4" : actColor(clamped);
      return { x, y, scale, color, clamped };
    });
  }, [activations, columns, isOutput, prediction]);

  return (
    <group position={[0, 0, z]}>
      {neurons.map((n, i) => (
        <mesh key={i} position={[n.x, n.y, 0]} scale={n.scale}>
          <sphereGeometry args={[1, 12, 12]} />
          <meshStandardMaterial color={n.color} emissive={n.color} emissiveIntensity={n.clamped * 0.6} roughness={0.4} metalness={0.1} />
        </mesh>
      ))}
    </group>
  );
}
