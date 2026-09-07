"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

function StarField() {
  const ref = useRef<THREE.Points>(null!);

  const positions = useMemo(() => {
    const pos = new Float32Array(3000 * 3);
    for (let i = 0; i < 3000; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 20;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 20;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 20;
    }
    return pos;
  }, []);

  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.x += delta * 0.02;
      ref.current.rotation.y += delta * 0.03;
    }
  });

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#6366f1"
        size={0.015}
        sizeAttenuation
        depthWrite={false}
        opacity={0.6}
      />
    </Points>
  );
}

function FloatingOrb({ position, color, speed }: { position: [number, number, number]; color: string; speed: number }) {
  const ref = useRef<THREE.Mesh>(null!);

  useFrame(({ clock }) => {
    if (ref.current) {
      ref.current.position.y = position[1] + Math.sin(clock.elapsedTime * speed) * 0.5;
      ref.current.position.x = position[0] + Math.cos(clock.elapsedTime * speed * 0.7) * 0.3;
    }
  });

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.15, 32, 32]} />
      <meshBasicMaterial color={color} transparent opacity={0.3} />
    </mesh>
  );
}

export default function Scene3D() {
  return (
    <div className="fixed inset-0 -z-10">
      <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
        <ambientLight intensity={0.1} />
        <StarField />
        <FloatingOrb position={[-2, 1, -3]} color="#6366f1" speed={0.8} />
        <FloatingOrb position={[2, -1, -2]} color="#8b5cf6" speed={1.2} />
        <FloatingOrb position={[0, 2, -4]} color="#a78bfa" speed={0.6} />
        <FloatingOrb position={[-3, -2, -5]} color="#4f46e5" speed={1.0} />
        <FloatingOrb position={[3, 0, -3]} color="#7c3aed" speed={0.9} />
      </Canvas>
    </div>
  );
}
