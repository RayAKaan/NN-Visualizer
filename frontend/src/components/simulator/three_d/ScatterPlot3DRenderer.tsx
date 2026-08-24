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
  new THREE.Color("#0072B2"),
  new THREE.Color("#E69F00"),
  new THREE.Color("#009E73"),
  new THREE.Color("#B45309"),
  new THREE.Color("#CC79A7"),
  new THREE.Color("#D55E00"),
  new THREE.Color("#56B4E9"),
  new THREE.Color("#00806A"),
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
