import React, { useMemo } from "react";

interface Props {
  activations: number[];
  z: number;
  isOutput?: boolean;
}

const NeuronLayer3D: React.FC<Props> = ({ activations, z, isOutput }) => {
  const positions = useMemo(() => {
    const cols = Math.ceil(Math.sqrt(activations.length));
    return activations.map((_, i) => {
      const x = (i % cols) * 0.6 - (cols * 0.6) / 2;
      const y = Math.floor(i / cols) * 0.6 - (cols * 0.6) / 2;
      return [x, y, z] as [number, number, number];
    });
  }, [activations, z]);

  return (
    <>
      {positions.map((pos, i) => {
        const activation = activations[i];
        const scale = 0.15 + activation * 0.45;

        return (
          <mesh key={i} position={pos} scale={scale}>
            <sphereGeometry args={[1, 16, 16]} />
            <meshStandardMaterial
              color={
                isOutput
                  ? "orange"
                  : activation > 0.6
                  ? "#7ae694"
                  : "#a162db"
              }
              emissiveIntensity={activation * 0.6}
            />
          </mesh>
        );
      })}
    </>
  );
};

export default NeuronLayer3D;
