import React, { useMemo } from "react";
import { Line } from "@react-three/drei";
import { Edge } from "../../types";

interface Props {
  edges: Edge[];
  zFrom: number;
  zTo: number;
}

const EdgeBundle3D: React.FC<Props> = ({ edges, zFrom, zTo }) => {
  const top = useMemo(
    () => [...edges].sort((a, b) => Math.abs(b.strength) - Math.abs(a.strength)).slice(0, 150),
    [edges]
  );

  return (
    <>
      {top.map((edge, i) => {
        const s = Math.abs(edge.strength);
        return (
          <Line
            key={i}
            points={[[edge.from * 0.05 - 2.5, 0, zFrom], [edge.to * 0.08 - 2.5, 0, zTo]]}
            color={edge.strength >= 0 ? "#06b6d4" : "#8b5cf6"}
            lineWidth={1 + s * 2}
            transparent
            opacity={Math.min(0.8, 0.1 + s * 0.6)}
          />
        );
      })}
    </>
  );
};

export default EdgeBundle3D;
