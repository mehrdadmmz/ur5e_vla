import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Mesh, Vector3 } from 'three';
import { Html } from '@react-three/drei';
import { Block } from '../../types';

interface BlockMeshProps {
  block: Block;
}

export function BlockMesh({ block }: BlockMeshProps) {
  const meshRef = useRef<Mesh>(null);
  const currentPos = useRef(
    new Vector3(block.position[0], block.position[2], block.position[1])
  );

  // Smooth position animation
  // Note: block.position[2] is the TOP of the block (ArUco marker position)
  // Three.js box center = top - half_height
  useFrame((_, delta) => {
    if (!meshRef.current) return;

    const target = new Vector3(
      block.position[0],
      block.position[2] - block.dimensions[2] / 2,
      block.position[1]
    );

    currentPos.current.lerp(target, Math.min(delta * 5, 1));
    meshRef.current.position.copy(currentPos.current);
  });

  // Determine opacity based on confidence
  const opacity = block.confident ? 1.0 : 0.6;

  return (
    <group>
      <mesh
        ref={meshRef}
        position={[
          block.position[0],
          block.position[2] - block.dimensions[2] / 2,
          block.position[1],
        ]}
        castShadow
        receiveShadow
      >
        <boxGeometry
          args={[block.dimensions[0], block.dimensions[2], block.dimensions[1]]}
        />
        <meshStandardMaterial
          color={block.color}
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
          block.position[1],
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
