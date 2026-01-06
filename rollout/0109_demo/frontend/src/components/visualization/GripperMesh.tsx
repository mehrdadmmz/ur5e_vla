import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Group, Vector3 } from 'three';

interface GripperMeshProps {
  position: [number, number, number];
  isOpen: boolean;
  holding: number | null;
}

export function GripperMesh({ position, isOpen, holding }: GripperMeshProps) {
  const groupRef = useRef<Group>(null);
  // Coordinate mapping (right-handed): Robot X→Three.js X, Robot Y→Three.js -Z, Robot Z→Three.js Y
  const currentPos = useRef(new Vector3(position[0], position[2], -position[1]));
  const fingerOffset = useRef(isOpen ? 0.025 : 0.01);

  useFrame((_, delta) => {
    if (!groupRef.current) return;

    // Smooth position interpolation (negate Y for right-handed coord system)
    const target = new Vector3(position[0], position[2], -position[1]);
    currentPos.current.lerp(target, Math.min(delta * 8, 1));
    groupRef.current.position.copy(currentPos.current);

    // Smooth finger animation
    const targetOffset = isOpen ? 0.025 : 0.01;
    fingerOffset.current += (targetOffset - fingerOffset.current) * Math.min(delta * 10, 1);
  });

  const fingerWidth = 0.008;
  const fingerLength = 0.04;
  const fingerHeight = 0.012;
  const bodyColor = holding !== null ? '#fbbf24' : '#6b7280';

  return (
    <group ref={groupRef} position={[position[0], position[2], -position[1]]}>
      {/* Gripper body */}
      <mesh position={[0, 0.035, 0]}>
        <cylinderGeometry args={[0.015, 0.02, 0.05]} />
        <meshStandardMaterial color={bodyColor} metalness={0.3} roughness={0.7} />
      </mesh>

      {/* Left finger */}
      <mesh position={[-fingerOffset.current, 0, 0]}>
        <boxGeometry args={[fingerWidth, fingerLength, fingerHeight]} />
        <meshStandardMaterial color="#9ca3af" metalness={0.4} roughness={0.6} />
      </mesh>

      {/* Right finger */}
      <mesh position={[fingerOffset.current, 0, 0]}>
        <boxGeometry args={[fingerWidth, fingerLength, fingerHeight]} />
        <meshStandardMaterial color="#9ca3af" metalness={0.4} roughness={0.6} />
      </mesh>
    </group>
  );
}
