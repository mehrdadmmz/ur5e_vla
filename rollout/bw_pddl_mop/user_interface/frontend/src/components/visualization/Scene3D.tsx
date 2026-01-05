import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import { useRobot } from '../../context/RobotContext';
import { BlockMesh } from './BlockMesh';
import { GripperMesh } from './GripperMesh';
import { CoordinateAxes } from './CoordinateAxes';

export function Scene3D() {
  const { state } = useRobot();

  return (
    <div className="w-full h-full min-h-[400px] bg-gray-100 rounded-lg overflow-hidden border border-gray-200 shadow-sm">
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0.8, 0.8, 0.8]} fov={50} />
        <OrbitControls
          target={[0, 0.05, 0.5]}
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

        {/* Coordinate axes */}
        <CoordinateAxes length={0.2} />

        {/* Table surface */}
        <mesh
          receiveShadow
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, -0.005, 0.5]}
        >
          <planeGeometry args={[0.8, 0.6]} />
          <meshStandardMaterial color="#e5e7eb" />
        </mesh>

        {/* Grid on table */}
        <Grid
          args={[0.8, 0.6]}
          position={[0, 0, 0.5]}
          cellSize={0.05}
          cellThickness={0.5}
          cellColor="#9ca3af"
          sectionSize={0.1}
          sectionThickness={1}
          sectionColor="#6b7280"
          fadeDistance={2}
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
