import React, { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

interface Props {
  z: number;
  activations: number[];
  columns: number;
  isOutput?: boolean;
  prediction?: number;
}

function actColor(a: number): THREE.Color {
  const c = Math.max(0, Math.min(1, a));
  if (c <= 0.5) {
    const t = c / 0.5;
    return new THREE.Color(`rgb(${Math.round(10 + t * 129)},${Math.round(14 + t * 78)},${Math.round(23 + t * 223)})`);
  }
  const t = (c - 0.5) / 0.5;
  return new THREE.Color(`rgb(${Math.round(139 + t * (6 - 139))},${Math.round(92 + t * 90)},${Math.round(246 + t * (212 - 246))})`);
}

type NeuronTarget = {
  x: number;
  y: number;
  scale: number;
  color: THREE.Color;
  emissive: number;
};

export default function NeuronLayer3D({ z, activations, columns, isOutput, prediction }: Props) {
  const spacing = 0.5;
  const refs = useRef<(THREE.Mesh | null)[]>([]);

  const neurons = useMemo<NeuronTarget[]>(() => {
    const rows = Math.ceil(activations.length / columns);
    return activations.map((a, i) => {
      const col = i % columns;
      const row = Math.floor(i / columns);
      const x = (col - columns / 2 + 0.5) * spacing;
      const y = -(row - rows / 2 + 0.5) * spacing;
      const clamped = Math.max(0, Math.min(1, a || 0));
      const scale = 0.12 + clamped * 0.33;
      const isWinner = !!isOutput && i === prediction;
      const color = isWinner ? new THREE.Color("#06b6d4") : actColor(clamped);
      return { x, y, scale, color, emissive: clamped * 0.65 + (isWinner ? 0.15 : 0) };
    });
  }, [activations, columns, isOutput, prediction]);

  useFrame((_, delta) => {
    const damp = 1 - Math.exp(-10 * delta);
    refs.current.forEach((mesh, i) => {
      if (!mesh || !neurons[i]) return;
      const target = neurons[i];
      mesh.position.x = THREE.MathUtils.lerp(mesh.position.x, target.x, damp);
      mesh.position.y = THREE.MathUtils.lerp(mesh.position.y, target.y, damp);
      const s = THREE.MathUtils.lerp(mesh.scale.x, target.scale, damp);
      mesh.scale.setScalar(s);
      const mat = mesh.material as THREE.MeshStandardMaterial;
      mat.color.lerp(target.color, damp);
      mat.emissive.lerp(target.color, damp);
      mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, target.emissive, damp);
    });
  });

  return (
    <group position={[0, 0, z]}>
      {neurons.map((n, i) => (
        <mesh
          key={i}
          ref={(el) => {
            refs.current[i] = el;
          }}
          position={[n.x, n.y, 0]}
          scale={n.scale}
        >
          <sphereGeometry args={[1, 12, 12]} />
          <meshStandardMaterial color={n.color} emissive={n.color} emissiveIntensity={n.emissive} roughness={0.35} metalness={0.18} />
        </mesh>
      ))}
    </group>
  );
}
