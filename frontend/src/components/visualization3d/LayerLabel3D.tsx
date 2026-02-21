import React from "react";
import { Html } from "@react-three/drei";

const LayerLabel3D: React.FC<{ position: [number, number, number]; label: string }> = ({ position, label }) => (
  <Html position={position} center>
    <div className="layer-label-3d">{label}</div>
  </Html>
);

export default LayerLabel3D;
