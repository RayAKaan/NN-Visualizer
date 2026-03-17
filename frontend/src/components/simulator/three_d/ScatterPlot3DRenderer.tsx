import React, { useMemo } from "react";
import * as THREE from "three";

interface ScatterPoint {
  coords: number[];
  label: number;
}

interface ScatterProps {
  points: ScatterPoint[];
}

const COLORS = [
  new THREE.Color("#38bdf8"),
  new THREE.Color("#f472b6"),
  new THREE.Color("#34d399"),
  new THREE.Color("#facc15"),
  new THREE.Color("#a78bfa"),
  new THREE.Color("#fb7185"),
  new THREE.Color("#f97316"),
  new THREE.Color("#60a5fa"),
];

export function ScatterPlot3DRenderer({ points }: ScatterProps) {
  const geometry = useMemo(() => {
    if (!points.length) return null;
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);

    points.forEach((p, i) => {
      const x = p.coords[0] ?? 0;
      const y = p.coords[1] ?? 0;
      const z = p.coords[2] ?? 0;
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
      const c = COLORS[p.label % COLORS.length];
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    });

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geom;
  }, [points]);

  if (!geometry) return null;
  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.06} vertexColors opacity={0.9} transparent />
    </points>
  );
}
