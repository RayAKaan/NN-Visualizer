import React from "react";
import * as THREE from "three";

interface Props {
  x: number;
  activations: number[];
  count: number;
  winner?: number;
}

function colorFor(a: number) {
  const c = Math.max(0, Math.min(1, a));
  if (c <= 0.5) {
    const t = c / 0.5;
    return new THREE.Color(`rgb(${Math.round(10 + t * 129)},${Math.round(14 + t * 78)},${Math.round(23 + t * 223)})`);
  }
  const t = (c - 0.5) / 0.5;
  return new THREE.Color(`rgb(${Math.round(139 + t * (6 - 139))},${Math.round(92 + t * 90)},${Math.round(246 + t * (212 - 246))})`);
}

export default function NeuronLayer3D({ x, activations, count, winner }: Props) {
  const size = Math.ceil(Math.sqrt(count));

  return (
    <group position={[x, 0, 0]}>
      {Array.from({ length: count }).map((_, i) => {
        const a = activations[i] ?? 0;
        const row = Math.floor(i / size);
        const col = i % size;
        const y = (row - size / 2) * 0.35;
        const z = (col - size / 2) * 0.35;
        const radius = 0.04 + Math.min(a * 0.08, 0.1);
        const color = winner === i ? new THREE.Color("#06b6d4") : colorFor(a);
        return (
          <mesh key={i} position={[0, y, z]}>
            <sphereGeometry args={[radius, 12, 12]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={Math.min(0.8, a * 0.7)} />
          </mesh>
        );
      })}
    </group>
  );
}
