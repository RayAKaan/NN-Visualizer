import React from "react";
import { Html } from "@react-three/drei";

interface Props {
  position: [number, number, number];
  text: string;
}

export default function LayerLabel3D({ position, text }: Props) {
  return (
    <Html
      position={position}
      center
      style={{
        color: "#94a3b8",
        fontSize: 11,
        fontFamily: "'JetBrains Mono', monospace",
        fontWeight: 600,
        whiteSpace: "nowrap",
        userSelect: "none",
        pointerEvents: "none",
        textShadow: "0 0 8px rgba(0,0,0,0.8)",
      }}
    >
      {text}
    </Html>
  );
}
