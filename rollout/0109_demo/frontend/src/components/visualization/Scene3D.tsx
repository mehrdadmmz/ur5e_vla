import { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Line } from '@react-three/drei';
import { useRobot } from '../../context/RobotContext';
import { BlockMesh } from './BlockMesh';
import { GripperMesh } from './GripperMesh';
import { CoordinateAxes } from './CoordinateAxes';

// Default values (will be overwritten by config fetch)
const DEFAULT_TABLE_Z = -0.15;
const DEFAULT_TABLE_CENTER_Y = 0.5;

interface SceneConfig {
  tableY: number;      // Robot Z -> Three.js Y
  tableCenterZ: number; // Robot Y -> Three.js Z
}

export function Scene3D() {
  const { state } = useRobot();
  const [config, setConfig] = useState<SceneConfig>({
    tableY: DEFAULT_TABLE_Z,
    tableCenterZ: DEFAULT_TABLE_CENTER_Y,
  });

  // Fetch config from backend on mount
  useEffect(() => {
    fetch('/api/system/config')
      .then((res) => res.json())
      .then((data) => {
        setConfig({
          tableY: data.table_z ?? DEFAULT_TABLE_Z,
          tableCenterZ: data.table_center_y ?? DEFAULT_TABLE_CENTER_Y,
        });
      })
      .catch((err) => {
        console.warn('Failed to fetch config, using defaults:', err);
      });
  }, []);

  const { tableY, tableCenterZ } = config;
  // Negate tableCenterZ for right-handed coordinate system (Robot Y → Three.js -Z)
  const tableZ = -tableCenterZ;

  return (
    <div className="w-full h-full min-h-[400px] bg-gray-100 rounded-lg overflow-hidden border border-gray-200 shadow-sm">
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0.8, 0.6, -0.8]} fov={50} />
        <OrbitControls
          target={[0, tableY + 0.05, tableZ]}
          minDistance={0.3}
          maxDistance={2}
          enablePan={true}
        />

        {/* Lighting */}
        <ambientLight intensity={0.6} />
        <directionalLight
          position={[2, 4, 2]}
          intensity={0.8}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />
        <pointLight position={[-2, 2, 2]} intensity={0.2} />

        {/* Coordinate axes at table surface */}
        <group position={[0, tableY, tableZ]}>
          <CoordinateAxes length={0.2} />
        </group>

        {/* Table surface */}
        <mesh
          receiveShadow
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, tableY - 0.005, tableZ]}
        >
          <planeGeometry args={[0.8, 0.6]} />
          <meshStandardMaterial color="#e5e7eb" />
        </mesh>

        {/* Grid on table */}
        <Grid
          args={[0.8, 0.6]}
          position={[0, tableY, tableZ]}
          cellSize={0.05}
          cellThickness={0.5}
          cellColor="#9ca3af"
          sectionSize={0.1}
          sectionThickness={1}
          sectionColor="#6b7280"
          fadeDistance={2}
        />

        {/* Table border (black line) */}
        <Line
          points={[
            [-0.4, tableY + 0.001, tableZ - 0.3],
            [0.4, tableY + 0.001, tableZ - 0.3],
            [0.4, tableY + 0.001, tableZ + 0.3],
            [-0.4, tableY + 0.001, tableZ + 0.3],
            [-0.4, tableY + 0.001, tableZ - 0.3],
          ]}
          color="black"
          lineWidth={2}
        />

        {/* Blocks */}
        {state.blocks.map((block) => (
          <BlockMesh key={block.id} block={block} />
        ))}

        {/* Gripper */}
        <GripperMesh
          position={state.gripper.position}
          isOpen={state.gripper.open}
          holding={state.gripper.holding}
        />
      </Canvas>
    </div>
  );
}
