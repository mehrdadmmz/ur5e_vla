import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh, Vector3 } from 'three';
import { Html } from '@react-three/drei';
import { Block } from '../../types';

interface BlockMeshProps {
  block: Block;
  isBridge?: boolean;
  isFloating?: boolean;
}

// Bridge plank color (purple)
const BRIDGE_COLOR = '#a855f7';
// Floating block color (yellow) - perception error
const FLOATING_COLOR = '#eab308';

export function BlockMesh({ block, isBridge = false, isFloating = false }: BlockMeshProps) {
  const meshRef = useRef<Mesh>(null);
  // Coordinate mapping (right-handed): Robot X→Three.js X, Robot Y→Three.js -Z, Robot Z→Three.js Y
  const currentPos = useRef(
    new Vector3(block.position[0], block.position[2], -block.position[1])
  );
  // Robot yaw (around Z) maps to Three.js rotation around Y
  const currentYaw = useRef(block.yaw ?? 0);

  // Smooth position and rotation animation
  // Note: block.position[2] is the TOP of the block (ArUco marker position)
  // Three.js box center = top - half_height
  useFrame((_, delta) => {
    if (!meshRef.current) return;

    const target = new Vector3(
      block.position[0],
      block.position[2] - block.dimensions[2] / 2,
      -block.position[1]  // Negate for right-handed coord system
    );

    currentPos.current.lerp(target, Math.min(delta * 5, 1));
    meshRef.current.position.copy(currentPos.current);

    // Lerp rotation (robot yaw around Z -> Three.js Y-rotation)
    const targetYaw = block.yaw ?? 0;
    currentYaw.current += (targetYaw - currentYaw.current) * Math.min(delta * 5, 1);
    meshRef.current.rotation.y = currentYaw.current;
  });

  // Determine opacity based on confidence
  const opacity = block.confident ? 1.0 : 0.6;

  // Use special colors for bridge planks and floating blocks, otherwise use normal block color
  const displayColor = isFloating ? FLOATING_COLOR : isBridge ? BRIDGE_COLOR : block.color;

  return (
    <group>
      <mesh
        ref={meshRef}
        position={[
          block.position[0],
          block.position[2] - block.dimensions[2] / 2,
          -block.position[1],  // Negate for right-handed coord system
        ]}
        castShadow
        receiveShadow
      >
        <boxGeometry
          args={[block.dimensions[0], block.dimensions[2], block.dimensions[1]]}
        />
        <meshStandardMaterial
          color={displayColor}
          metalness={0.1}
          roughness={0.8}
          transparent={!block.confident}
          opacity={opacity}
        />
      </mesh>

      {/* Block ID label - positioned just above the block top */}
      <Html
        position={[
          block.position[0],
          block.position[2] + 0.02,
          -block.position[1],  // Negate for right-handed coord system
        ]}
        center
        distanceFactor={0.5}
      >
        <div
          className={`px-1.5 py-0.5 rounded text-xs font-bold ${
            block.stale
              ? 'bg-red-600 text-white'
              : 'bg-gray-800 text-white border border-gray-600'
          }`}
        >
          {block.id}
        </div>
      </Html>
    </group>
  );
}
