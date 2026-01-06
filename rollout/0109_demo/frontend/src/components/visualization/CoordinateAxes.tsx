interface CoordinateAxesProps {
  length?: number;
}

export function CoordinateAxes({ length = 0.2 }: CoordinateAxesProps) {
  return (
    <group>
      {/* X axis - Red */}
      <mesh position={[length / 2, 0, 0]}>
        <cylinderGeometry args={[0.003, 0.003, length, 8]} />
        <meshStandardMaterial color="#ef4444" />
      </mesh>
      <mesh rotation={[0, 0, -Math.PI / 2]} position={[length / 2, 0, 0]}>
        <cylinderGeometry args={[0.003, 0.003, length, 8]} />
        <meshStandardMaterial color="#ef4444" />
      </mesh>

      {/* Y axis - Green */}
      <mesh position={[0, 0, length / 2]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.003, 0.003, length, 8]} />
        <meshStandardMaterial color="#22c55e" />
      </mesh>

      {/* Z axis - Blue */}
      <mesh position={[0, length / 2, 0]}>
        <cylinderGeometry args={[0.003, 0.003, length, 8]} />
        <meshStandardMaterial color="#3b82f6" />
      </mesh>

      {/* Origin sphere */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.008, 16, 16]} />
        <meshStandardMaterial color="#ffffff" />
      </mesh>
    </group>
  );
}
