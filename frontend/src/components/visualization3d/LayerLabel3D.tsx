import React from "react";
import { Text } from "@react-three/drei";

interface Props {
  x: number;
  label: string;
}

export default function LayerLabel3D({ x, label }: Props) {
  return (
    <Text position={[x, 2.2, 0]} color="#94a3b8" fontSize={0.18} anchorX="center" anchorY="middle">
      {label}
    </Text>
  );
}
