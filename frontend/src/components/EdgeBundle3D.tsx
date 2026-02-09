import React, { useMemo } from "react";
import * as THREE from "three";
import { QuadraticBezierLine } from "@react-three/drei";
import { Edge } from "../types/NeuralState";

interface Props {
  edges: Edge[];
  zFrom: number;
  zTo: number;
}

/* =====================================================
   Visual tuning constants

const MIN_STRENGTH = 0.05;
const MAX_WIDTH = 2.4;
const BASE_WIDTH = 0.45;

const POSITIVE_COLOR = new THREE.Color("#7ae694"); // mint green
const NEGATIVE_COLOR = new THREE.Color("#a162db"); // violet

/* =====================================================
   Component

const EdgeBundle3D: React.FC<Props> = ({ edges, zFrom, zTo }) => {
  const lines = useMemo(() => {
    return edges
      .map((edge, i) => {
        const strength = Math.abs(edge.strength);
        if (strength < MIN_STRENGTH) return null;

        /* --- geometry --- */
        const x1 = edge.from * 0.035 - 2;
        const x2 = edge.to * 0.085 - 2;

        const start: [number, number, number] = [x1, 0, zFrom];
        const end: [number, number, number] = [x2, 0, zTo];

        /* --- smooth depth curve --- */
        const mid: [number, number, number] = [
          (x1 + x2) * 0.5,
          Math.min(0.9, strength * 1.4), // vertical lift
          (zFrom + zTo) * 0.5,
        ];

        /* --- visual encoding --- */
        const width = Math.min(
          MAX_WIDTH,
          BASE_WIDTH + strength * 2.2
        );

        const opacity = Math.min(
          0.85,
          0.18 + strength * 0.6
        );

        const color =
          edge.strength > 0
            ? POSITIVE_COLOR
            : NEGATIVE_COLOR;

        return (
          <group key={`edge-${i}`}>
            {/* glow layer */}
            <QuadraticBezierLine
              start={start}
              end={end}
              mid={mid}
              color={color}
              lineWidth={width * 1.6}
              transparent
              opacity={opacity * 0.18}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />

            {/* core signal */}
            <QuadraticBezierLine
              start={start}
              end={end}
              mid={mid}
              color={color}
              lineWidth={width}
              transparent
              opacity={opacity}
              depthWrite={false}
              blending={THREE.AdditiveBlending}
            />
          </group>
        );
      })
      .filter(Boolean);
  }, [edges, zFrom, zTo]);

  return <>{lines}</>;
};

export default React.memo(EdgeBundle3D);
