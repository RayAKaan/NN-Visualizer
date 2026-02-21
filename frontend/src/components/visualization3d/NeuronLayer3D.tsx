import React, { useMemo } from "react";

interface Props {
  activations: number[];
  z: number;
}

const NeuronLayer3D: React.FC<Props> = ({ activations, z }) => {
  const positions = useMemo(() => {
    const cols = Math.ceil(Math.sqrt(activations.length));
    return activations.map((_, i) => [
      (i % cols) * 0.55 - (cols * 0.55) / 2,
      Math.floor(i / cols) * 0.55 - (cols * 0.55) / 2,
      z,
    ] as [number, number, number]);
  }, [activations, z]);

  return (
    <>
      {positions.map((pos, i) => {
        const a = Math.max(0, Math.min(1, activations[i] ?? 0));
        return (
          <mesh key={i} position={pos} scale={0.15 + a * 0.35}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshStandardMaterial color={a > 0.5 ? "#06b6d4" : "#8b5cf6"} emissive={a > 0.5 ? "#06b6d4" : "#8b5cf6"} emissiveIntensity={a * 0.8} />
          </mesh>
        );
      })}
    </>
  );
};

export default NeuronLayer3D;
