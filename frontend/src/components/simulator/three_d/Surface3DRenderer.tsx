import React, { useMemo } from "react";
import * as THREE from "three";
import { GRADIENT_RAMP_DARK, sampleRamp } from "../../../utils/colorRamps";

interface SurfaceProps {
  gridX: number[];
  gridY: number[];
  lossSurface: number[][];
  heightScale?: number;
}

export function Surface3DRenderer({ gridX, gridY, lossSurface, heightScale = 1.0 }: SurfaceProps) {
  const geometry = useMemo(() => {
    const rows = gridY.length;
    const cols = gridX.length;
    if (!rows || !cols) return null;

    const positions = new Float32Array(rows * cols * 3);
    const colors = new Float32Array(rows * cols * 3);
    const indices: number[] = [];

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        const v = lossSurface[i][j];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    const span = max - min || 1;

    let idx = 0;
    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        const x = gridX[j];
        const y = gridY[i];
        const z = (lossSurface[i][j] - min) / span;
        positions[idx] = x;
        positions[idx + 1] = y;
        positions[idx + 2] = z * heightScale;

        const [r, g, b] = sampleRamp(GRADIENT_RAMP_DARK, z);
        colors[idx] = r / 255;
        colors[idx + 1] = g / 255;
        colors[idx + 2] = b / 255;
        idx += 3;
      }
    }

    for (let i = 0; i < rows - 1; i += 1) {
      for (let j = 0; j < cols - 1; j += 1) {
        const a = i * cols + j;
        const b = i * cols + j + 1;
        const c = (i + 1) * cols + j;
        const d = (i + 1) * cols + j + 1;
        indices.push(a, b, d, a, d, c);
      }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.setIndex(indices);
    geom.computeVertexNormals();
    return geom;
  }, [gridX, gridY, lossSurface, heightScale]);

  if (!geometry) return null;
  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2.5, 0, 0]}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} roughness={0.5} metalness={0.1} />
    </mesh>
  );
}
