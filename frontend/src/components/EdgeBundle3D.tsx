import React from "react";
import { Line } from "@react-three/drei";
import { Edge } from "../types/NeuralState";

interface Props {
  edges: Edge[];
  zFrom: number;
  zTo: number;
  highlight?: number;
}

const EdgeBundle3D: React.FC<Props> = ({ edges, zFrom, zTo }) => {
  return (
    <>
      {edges.map((edge, i) => {
        const strength = Math.abs(edge.strength);
        const color = edge.strength > 0 ? "#7ae694" : "#a162db";

        return (
          <Line
            key={i}
            points={[
              [edge.from * 0.03 - 2, 0, zFrom],
              [edge.to * 0.08 - 2, 0, zTo],
            ]}
            color={color}
            lineWidth={1 + strength * 2}
            transparent
            opacity={0.15 + strength * 0.6}
          />
        );
      })}
    </>
  );
};

export default EdgeBundle3D;
