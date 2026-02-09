import React, { useMemo, useRef, useLayoutEffect } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";

interface Props {
  activations: number[];
  z: number;
  isOutput?: boolean;
}

/* =====================================================
   Configuration

const CONFIG = {
  gridGap: 0.6,
  baseScale: 0.2,
  activationScaleFactor: 0.5,
  colors: {
    idle: new THREE.Color("#2a1d3d"),
    active: new THREE.Color("#4ff0b7"),
    output: new THREE.Color("#ff7b00"),
    high: new THREE.Color("#ffffff"),
  },
};

/* =====================================================
   Component

const NeuronLayer3D: React.FC<Props> = ({
  activations,
  z,
  isOutput,
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);

  const layout = useMemo(() => {
    const count = activations.length;
    const cols = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / cols);
    const width = (cols - 1) * CONFIG.gridGap;
    const height = (rows - 1) * CONFIG.gridGap;
    return { cols, width, height };
  }, [activations.length]);

  useLayoutEffect(() => {
    if (!meshRef.current) return;

    activations.forEach((aRaw, i) => {
      const a = THREE.MathUtils.clamp(aRaw, 0, 1);

      const col = i % layout.cols;
      const row = Math.floor(i / layout.cols);

      const x = col * CONFIG.gridGap - layout.width / 2;
      const y = -(row * CONFIG.gridGap - layout.height / 2);

      const eased = a * a;
      const scale =
        CONFIG.baseScale +
        eased * CONFIG.activationScaleFactor;

      dummy.position.set(x, y, z);
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);

      if (isOutput) {
        color
          .copy(CONFIG.colors.output)
          .lerp(CONFIG.colors.high, a * 0.5);
      } else {
        color
          .copy(CONFIG.colors.idle)
          .lerp(CONFIG.colors.active, a);
        if (a > 0.8) {
          color.lerp(CONFIG.colors.high, (a - 0.8) * 2);
        }
      }

      meshRef.current!.setColorAt(i, color);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor)
      meshRef.current.instanceColor.needsUpdate = true;
  }, [activations, z, isOutput, layout, dummy, color]);

  // subtle organic motion (cheap)
  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const t = clock.getElapsedTime();
    meshRef.current.position.y =
      Math.sin(t * 0.4 + z) * 0.08;
    meshRef.current.rotation.z =
      Math.sin(t * 0.2 + z) * 0.015;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, activations.length]}
      frustumCulled={false}
    >
      <sphereGeometry args={[1, 24, 24]} />
      <meshPhysicalMaterial
        vertexColors
        roughness={0.25}
        metalness={0.15}
        transmission={0.2}
        thickness={1.2}
        clearcoat={1}
        clearcoatRoughness={0.1}
        emissive={isOutput ? "#ff4400" : "#220033"}
        emissiveIntensity={isOutput ? 0.45 : 0.25}
        toneMapped={false}
      />
    </instancedMesh>
  );
};

export default React.memo(NeuronLayer3D);
