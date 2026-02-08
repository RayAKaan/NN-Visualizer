import React, { useMemo, useRef, useLayoutEffect } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  QuadraticBezierLine,
  PerspectiveCamera,
  Environment,
} from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";

/* =====================================================
   Types
===================================================== */

interface Props {
  hidden1: number[];
  hidden2: number[];
  output: number[];
  weightsHidden1Hidden2: number[][] | null;
  weightsHidden2Output: number[][] | null;
}

interface NodeData {
  pos: THREE.Vector3;
  activation: number;
}

/* =====================================================
   Configuration
===================================================== */

const CONFIG = {
  grid: {
    gap: 0.8,
    zSpacing: 6,
  },
  colors: {
    idle: new THREE.Color("#1a1a2e"),
    low: new THREE.Color("#4b2a63"),
    high: new THREE.Color("#4ff0b7"),
    output: new THREE.Color("#ff9900"),
    negative: new THREE.Color("#bd93f9"),
  },
  edges: {
    minStrength: 0.06,
    maxCount: 220,
  },
};

/* =====================================================
   Helpers
===================================================== */

const calculateGridPositions = (
  activations: number[],
  z: number
): NodeData[] => {
  const count = activations.length;
  const cols = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / cols);

  const width = (cols - 1) * CONFIG.grid.gap;
  const height = (rows - 1) * CONFIG.grid.gap;

  return activations.map((a, i) => ({
    activation: a,
    pos: new THREE.Vector3(
      (i % cols) * CONFIG.grid.gap - width / 2,
      -(Math.floor(i / cols) * CONFIG.grid.gap - height / 2),
      z
    ),
  }));
};

/* =====================================================
   Neuron Layer (Instanced)
===================================================== */

const NeuronLayer = ({
  nodes,
  isOutput = false,
}: {
  nodes: NodeData[];
  isOutput?: boolean;
}) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const color = useMemo(() => new THREE.Color(), []);

  useLayoutEffect(() => {
    if (!meshRef.current) return;

    nodes.forEach((n, i) => {
      const a = THREE.MathUtils.clamp(n.activation, 0, 1);
      const scale = 0.7 + Math.pow(a, 1.6) * 0.55;

      dummy.position.copy(n.pos);
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);

      if (isOutput) {
        color.copy(CONFIG.colors.output).lerp(new THREE.Color("#ffffff"), a * 0.6);
      } else {
        color
          .copy(CONFIG.colors.idle)
          .lerp(CONFIG.colors.low, a)
          .lerp(CONFIG.colors.high, Math.max(0, a - 0.55) * 2);
      }

      meshRef.current!.setColorAt(i, color);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
    meshRef.current.instanceColor!.needsUpdate = true;
  }, [nodes, isOutput]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, nodes.length]}>
      <sphereGeometry args={[0.22, 24, 24]} />
      <meshStandardMaterial
        toneMapped={false}
        emissive="white"
        emissiveIntensity={0.35}
        roughness={0.3}
        metalness={0.55}
      />
    </instancedMesh>
  );
};

/* =====================================================
   Animated Connection
===================================================== */

const AnimatedConnection = ({
  start,
  end,
  strength,
  color,
}: {
  start: THREE.Vector3;
  end: THREE.Vector3;
  strength: number;
  color: THREE.Color;
}) => {
  const dashOffset = useRef(0);

  useFrame((_, dt) => {
    dashOffset.current -= dt * (0.5 + strength * 3);
  });

  const mid: [number, number, number] = [
    (start.x + end.x) * 0.5,
    (start.y + end.y) * 0.5 + strength * 1.2,
    (start.z + end.z) * 0.5,
  ];

  return (
    <QuadraticBezierLine
      start={start}
      end={end}
      mid={mid}
      color={color}
      lineWidth={0.5 + strength * 2.2}
      dashed
      dashOffset={dashOffset.current}
      dashSize={1.1}
      gapSize={1.1}
      transparent
      opacity={0.7}
      depthWrite={false}
    />
  );
};

/* =====================================================
   Active Connections
===================================================== */

const ActiveConnections = ({
  from,
  to,
  weights,
}: {
  from: NodeData[];
  to: NodeData[];
  weights: number[][];
}) => {
  let rendered = 0;

  return (
    <>
      {weights.map((row, i) =>
        row.map((w, j) => {
          if (rendered >= CONFIG.edges.maxCount) return null;

          const src = from[i];
          const dst = to[j];
          if (!src || !dst) return null;

          const signal = w * src.activation;
          const mag = Math.abs(signal);
          if (mag < CONFIG.edges.minStrength) return null;

          rendered++;

          return (
            <AnimatedConnection
              key={`${i}-${j}`}
              start={src.pos}
              end={dst.pos}
              strength={mag}
              color={signal > 0 ? CONFIG.colors.high : CONFIG.colors.negative}
            />
          );
        })
      )}
    </>
  );
};

/* =====================================================
   Main Component
===================================================== */

const Network3D: React.FC<Props> = ({
  hidden1,
  hidden2,
  output,
  weightsHidden1Hidden2,
  weightsHidden2Output,
}) => {
  const h1 = useMemo(
    () => calculateGridPositions(hidden1, -CONFIG.grid.zSpacing),
    [hidden1]
  );
  const h2 = useMemo(
    () => calculateGridPositions(hidden2, 0),
    [hidden2]
  );
  const out = useMemo(
    () => calculateGridPositions(output, CONFIG.grid.zSpacing),
    [output]
  );

  return (
    <div
      style={{
        height: 600,
        background: "#06070d",
        borderRadius: 14,
        overflow: "hidden",
      }}
    >
      <Canvas dpr={[1, 2]}>
        <PerspectiveCamera makeDefault position={[9, 6, 11]} fov={45} />
        <OrbitControls
          minDistance={7}
          maxDistance={20}
          enablePan={false}
          dampingFactor={0.08}
        />

        {/* Lighting */}
        <ambientLight intensity={0.35} />
        <directionalLight position={[6, 10, 6]} intensity={1.0} />
        <pointLight position={[-6, -4, -6]} intensity={0.6} />

        {/* Layers */}
        <NeuronLayer nodes={h1} />
        <NeuronLayer nodes={h2} />
        <NeuronLayer nodes={out} isOutput />

        {/* Connections */}
        {weightsHidden1Hidden2 && (
          <ActiveConnections from={h1} to={h2} weights={weightsHidden1Hidden2} />
        )}
        {weightsHidden2Output && (
          <ActiveConnections from={h2} to={out} weights={weightsHidden2Output} />
        )}

        {/* Effects */}
        <EffectComposer>
          <Bloom luminanceThreshold={0.35} intensity={1.25} />
          <Vignette darkness={0.45} />
        </EffectComposer>

        <Environment preset="night" />
      </Canvas>
    </div>
  );
};

export default React.memo(Network3D);