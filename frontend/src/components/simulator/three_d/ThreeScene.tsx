import React from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

export function ThreeScene({ children, height = 360 }: { children?: React.ReactNode; height?: number }) {
  return (
    <div style={{ height }} className="three-scene">
      <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[4, 4, 4]} intensity={0.8} />
        {children}
        <OrbitControls enablePan enableZoom enableRotate />
      </Canvas>
    </div>
  );
}
