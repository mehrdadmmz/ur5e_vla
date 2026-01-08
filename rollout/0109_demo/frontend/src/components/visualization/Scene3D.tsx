import { useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid, Line } from '@react-three/drei';
import { DoubleSide, Shape, ShapeGeometry } from 'three';
import { useRobot } from '../../context/RobotContext';
import { BlockMesh } from './BlockMesh';
import { GripperMesh } from './GripperMesh';
import { CoordinateAxes } from './CoordinateAxes';
import { Block } from '../../types';

// Create a rounded rectangle shape for release zone visualization
function createRoundedRectShape(width: number, length: number, radius: number): Shape {
  const shape = new Shape();
  const w = width / 2;
  const l = length / 2;
  const r = Math.min(radius, w, l); // Clamp radius to avoid invalid shapes

  shape.moveTo(-w + r, -l);
  shape.lineTo(w - r, -l);
  shape.quadraticCurveTo(w, -l, w, -l + r);
  shape.lineTo(w, l - r);
  shape.quadraticCurveTo(w, l, w - r, l);
  shape.lineTo(-w + r, l);
  shape.quadraticCurveTo(-w, l, -w, l - r);
  shape.lineTo(-w, -l + r);
  shape.quadraticCurveTo(-w, -l, -w + r, -l);

  return shape;
}

// Default values (will be overwritten by config fetch)
const DEFAULT_TABLE_Z = -0.15;
const DEFAULT_TABLE_BOUNDS = { x_min: -0.3, x_max: 0.3, y_min: 0.3, y_max: 0.7 };
const DEFAULT_RELEASE_MIN_DIST = 0.15;

interface TableBounds {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
}

interface SceneConfig {
  tableY: number;      // Robot Z -> Three.js Y (table surface height)
  tableBounds: TableBounds;  // Robot X/Y bounds for table workspace
  releaseMinDist: number;  // Minimum distance from blocks for release action
}

// Get block center position in Three.js coords (accounting for height)
function getBlockCenter(block: Block): [number, number, number] {
  return [
    block.position[0],
    block.position[2] - block.dimensions[2] / 2,
    -block.position[1]
  ];
}

export function Scene3D() {
  const { state } = useRobot();
  const [config, setConfig] = useState<SceneConfig>({
    tableY: DEFAULT_TABLE_Z,
    tableBounds: DEFAULT_TABLE_BOUNDS,
    releaseMinDist: DEFAULT_RELEASE_MIN_DIST,
  });
  const [showInflation, setShowInflation] = useState(false);

  // Fetch config from backend on mount
  useEffect(() => {
    fetch('/api/system/config')
      .then((res) => res.json())
      .then((data) => {
        setConfig({
          tableY: data.table_z ?? DEFAULT_TABLE_Z,
          tableBounds: data.table_bounds ?? DEFAULT_TABLE_BOUNDS,
          releaseMinDist: data.release_min_dist ?? DEFAULT_RELEASE_MIN_DIST,
        });
      })
      .catch((err) => {
        console.warn('Failed to fetch config, using defaults:', err);
      });
  }, []);

  const { tableY, tableBounds, releaseMinDist } = config;

  // Calculate table dimensions and center from bounds
  // Robot coords: X (left/right), Y (forward/back), Z (up/down)
  // Three.js coords: X (same), Y (up/down), Z (forward/back, negated)
  const tableWidth = tableBounds.x_max - tableBounds.x_min;  // Robot X range
  const tableDepth = tableBounds.y_max - tableBounds.y_min;  // Robot Y range
  const tableCenterX = (tableBounds.x_min + tableBounds.x_max) / 2;
  const tableCenterY = (tableBounds.y_min + tableBounds.y_max) / 2;
  // Convert robot Y to Three.js Z (negated)
  const tableZ = -tableCenterY;

  // Extract relationship data from predicates
  const relationships = useMemo(() => {
    const blockMap = new Map(state.blocks.map(b => [String(b.id), b]));

    const beside: Array<{ from: Block; to: Block }> = [];
    const above: Array<{ top: Block; bottom: Block }> = [];
    const aboveBoth: Array<{ plank: Block; support1: Block; support2: Block }> = [];
    const seenBeside = new Set<string>(); // Avoid duplicate lines

    for (const pred of state.predicates) {
      if (pred.name === 'beside') {
        const key = [pred.args[0], pred.args[1]].sort().join('-');
        if (!seenBeside.has(key)) {
          const b1 = blockMap.get(pred.args[0]);
          const b2 = blockMap.get(pred.args[1]);
          if (b1 && b2) {
            beside.push({ from: b1, to: b2 });
            seenBeside.add(key);
          }
        }
      } else if (pred.name === 'above') {
        const top = blockMap.get(pred.args[0]);
        const bottom = blockMap.get(pred.args[1]);
        if (top && bottom) {
          above.push({ top, bottom });
        }
      } else if (pred.name === 'above_both') {
        const plank = blockMap.get(pred.args[0]);
        const s1 = blockMap.get(pred.args[1]);
        const s2 = blockMap.get(pred.args[2]);
        if (plank && s1 && s2) {
          aboveBoth.push({ plank, support1: s1, support2: s2 });
        }
      }
    }

    return { beside, above, aboveBoth };
  }, [state.predicates, state.blocks]);

  // Get IDs of bridge planks for special coloring
  const bridgePlankIds = useMemo(() => {
    return new Set(relationships.aboveBoth.map(r => r.plank.id));
  }, [relationships.aboveBoth]);

  // Get IDs of floating blocks (perception errors)
  const floatingBlockIds = useMemo(() => {
    const ids = new Set<number>();
    for (const pred of state.predicates) {
      if (pred.name === 'floating') {
        ids.add(Number(pred.args[0]));
      }
    }
    return ids;
  }, [state.predicates]);

  return (
    <div className="w-full h-full min-h-[400px] bg-gray-100 rounded-lg overflow-hidden border border-gray-200 shadow-sm relative">
      {/* Toggle for inflation visualization */}
      <div className="absolute top-2 right-2 z-10 bg-white/90 px-2 py-1 rounded text-xs flex items-center gap-2">
        <label className="flex items-center gap-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showInflation}
            onChange={(e) => setShowInflation(e.target.checked)}
            className="w-3 h-3"
          />
          <span>Release zones</span>
        </label>
      </div>
      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[0.6, 0.5, 0.1]} fov={50} />
        <OrbitControls
          target={[tableCenterX, tableY + 0.05, tableZ]}
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

        {/* Coordinate axes at table corner (robot origin: x_min, y_min) */}
        {/* This shows axes at the near-left corner so table extends in +X and +Y directions */}
        <group position={[tableBounds.x_min, tableY, -tableBounds.y_min]}>
          <CoordinateAxes length={0.2} />
        </group>

        {/* Table surface */}
        <mesh
          receiveShadow
          rotation={[-Math.PI / 2, 0, 0]}
          position={[tableCenterX, tableY - 0.005, tableZ]}
        >
          <planeGeometry args={[tableWidth, tableDepth]} />
          <meshStandardMaterial color="#e5e7eb" />
        </mesh>

        {/* Grid on table */}
        <Grid
          args={[tableWidth, tableDepth]}
          position={[tableCenterX, tableY, tableZ]}
          cellSize={0.05}
          cellThickness={0.5}
          cellColor="#9ca3af"
          sectionSize={0.1}
          sectionThickness={1}
          sectionColor="#6b7280"
          fadeDistance={2}
        />

        {/* Table border (black line) - using table_bounds */}
        <Line
          points={[
            [tableBounds.x_min, tableY + 0.001, -tableBounds.y_min],
            [tableBounds.x_max, tableY + 0.001, -tableBounds.y_min],
            [tableBounds.x_max, tableY + 0.001, -tableBounds.y_max],
            [tableBounds.x_min, tableY + 0.001, -tableBounds.y_max],
            [tableBounds.x_min, tableY + 0.001, -tableBounds.y_min],
          ]}
          color="black"
          lineWidth={2}
        />

        {/* Blocks */}
        {state.blocks.map((block) => (
          <BlockMesh
            key={block.id}
            block={block}
            isBridge={bridgePlankIds.has(block.id)}
            isFloating={floatingBlockIds.has(block.id)}
          />
        ))}

        {/* Rounded rectangle release zones around blocks */}
        {showInflation && state.blocks.map((block) => {
          // Zone = block shape + inflation radius on all sides (1/4 of releaseMinDist)
          const inflation = releaseMinDist / 4;
          const zoneWidth = block.dimensions[0] + inflation * 2;
          const zoneLength = block.dimensions[1] + inflation * 2;
          const cornerRadius = inflation;
          const yaw = block.yaw ?? 0;

          const shape = createRoundedRectShape(zoneWidth, zoneLength, cornerRadius);
          const geometry = new ShapeGeometry(shape);

          return (
            <mesh
              key={`inflation-${block.id}`}
              position={[block.position[0], tableY + 0.002, -block.position[1]]}
              rotation={[-Math.PI / 2, 0, yaw]}
              geometry={geometry}
            >
              <meshBasicMaterial color="#ef4444" transparent opacity={0.2} side={DoubleSide} />
            </mesh>
          );
        })}

        {/* Beside relationships - orange horizontal lines */}
        {relationships.beside.map(({ from, to }, idx) => {
          const fromPos = getBlockCenter(from);
          const toPos = getBlockCenter(to);
          return (
            <Line
              key={`beside-${idx}`}
              points={[fromPos, toPos]}
              color="#f97316"
              lineWidth={3}
              dashed
              dashSize={0.02}
              gapSize={0.01}
            />
          );
        })}

        {/* Above relationships - cyan vertical arrows */}
        {relationships.above.map(({ top, bottom }, idx) => {
          const topPos = getBlockCenter(top);
          const bottomPos = getBlockCenter(bottom);
          // Draw line from bottom block top to top block bottom
          const startY = bottom.position[2]; // top of bottom block
          const endY = top.position[2] - top.dimensions[2]; // bottom of top block
          return (
            <group key={`above-${idx}`}>
              <Line
                points={[
                  [bottomPos[0], startY, bottomPos[2]],
                  [topPos[0], endY, topPos[2]]
                ]}
                color="#06b6d4"
                lineWidth={2}
              />
              {/* Arrow head */}
              <mesh position={[topPos[0], endY, topPos[2]]}>
                <coneGeometry args={[0.008, 0.02, 8]} />
                <meshBasicMaterial color="#06b6d4" />
              </mesh>
            </group>
          );
        })}

        {/* Above_both (bridge) - purple lines to supports */}
        {relationships.aboveBoth.map(({ plank, support1, support2 }, idx) => {
          const plankPos = getBlockCenter(plank);
          const s1Pos = getBlockCenter(support1);
          const s2Pos = getBlockCenter(support2);
          const plankBottom = plank.position[2] - plank.dimensions[2];
          const s1Top = support1.position[2];
          const s2Top = support2.position[2];
          return (
            <group key={`bridge-${idx}`}>
              <Line
                points={[
                  [s1Pos[0], s1Top, s1Pos[2]],
                  [plankPos[0], plankBottom, plankPos[2]]
                ]}
                color="#a855f7"
                lineWidth={2}
              />
              <Line
                points={[
                  [s2Pos[0], s2Top, s2Pos[2]],
                  [plankPos[0], plankBottom, plankPos[2]]
                ]}
                color="#a855f7"
                lineWidth={2}
              />
            </group>
          );
        })}

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
