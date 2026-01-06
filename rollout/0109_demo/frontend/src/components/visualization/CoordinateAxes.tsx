import { Html } from '@react-three/drei';

interface CoordinateAxesProps {
  length?: number;
}

export function CoordinateAxes({ length = 0.2 }: CoordinateAxesProps) {
  return (
    <group>
      {/* X axis - Red (Robot X direction) */}
      <mesh rotation={[0, 0, -Math.PI / 2]} position={[length / 2, 0, 0]}>
        <cylinderGeometry args={[0.003, 0.003, length, 8]} />
        <meshStandardMaterial color="#ef4444" />
      </mesh>
      <Html position={[length + 0.02, 0, 0]} center>
        <div className="text-red-500 font-bold text-xs bg-white/80 px-1 rounded">X</div>
      </Html>

      {/* Y axis - Green (Robot Y direction, shown as Three.js -Z for right-handed) */}
      <mesh position={[0, 0, -length / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.003, 0.003, length, 8]} />
        <meshStandardMaterial color="#22c55e" />
      </mesh>
      <Html position={[0, 0, -(length + 0.02)]} center>
        <div className="text-green-500 font-bold text-xs bg-white/80 px-1 rounded">Y</div>
      </Html>

      {/* Z axis - Blue (Robot Z direction, shown as Three.js Y) */}
      <mesh position={[0, length / 2, 0]}>
        <cylinderGeometry args={[0.003, 0.003, length, 8]} />
        <meshStandardMaterial color="#3b82f6" />
      </mesh>
      <Html position={[0, length + 0.02, 0]} center>
        <div className="text-blue-500 font-bold text-xs bg-white/80 px-1 rounded">Z</div>
      </Html>

      {/* Origin sphere */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.008, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
    </group>
  );
}
