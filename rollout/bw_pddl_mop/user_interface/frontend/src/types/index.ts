// Block types
export type BlockClass = 2 | 3; // 2=cube, 3=plank

export interface BlockObservation {
  id: number;
  class: BlockClass;
  position: [number, number, number];
  dimensions: [number, number, number];
  confident: boolean;
  stale: boolean;
}

export type BlockState = 'on-table' | 'stacked' | 'held' | 'moving';

export interface Block extends BlockObservation {
  state: BlockState;
  color: string;
}

// PDDL types
export interface Predicate {
  name: string;
  args: string[];
}

export type ActionName =
  | 'pick-up'
  | 'put-down'
  | 'align'
  | 'cover'
  | 'unstack'
  | 'remove-beside'
  | 'release';

export interface PlanAction {
  name: ActionName | string;
  args: string[];
  status: 'pending' | 'executing' | 'completed' | 'failed';
}

export type GoalType = 'bridge' | 'tall_bridge' | 'tower' | 'cn_tower' | 'house';

export interface GoalInfo {
  name: GoalType;
  description: string;
  predicates: Predicate[];
}

// Robot types
export interface GripperState {
  position: [number, number, number];
  open: boolean;
  holding: number | null;
}

export type ExecutionStatus =
  | 'idle'
  | 'observing'
  | 'planning'
  | 'executing'
  | 'paused'
  | 'error'
  | 'completed';

// WebSocket message types
export interface WSStateUpdate {
  type: 'state_update';
  data: {
    execution_status: ExecutionStatus;
    predicates: Predicate[];
    goal: Predicate[];
    goal_name: string;
    goal_satisfied: boolean;
    unsatisfied_goals: Predicate[];
    blocks: BlockObservation[];
    current_plan: PlanAction[];
    current_action_index: number;
    cycle: number;
    max_cycles: number;
    gripper_position: [number, number, number];
    gripper_open: boolean;
    holding_block: number | null;
    robot_connected: boolean;
    camera_connected: boolean;
    last_update: number;
  };
}

export interface WSLogMessage {
  type: 'log';
  data: {
    level: 'debug' | 'info' | 'warn' | 'error';
    source: string;
    message: string;
    timestamp: string;
  };
}

export interface WSCommandResponse {
  type: 'command_response';
  command: string;
  success: boolean;
  message: string;
}

export type WSMessage = WSStateUpdate | WSLogMessage | WSCommandResponse | { type: string; data?: unknown };

// Log entry type
export interface LogEntry {
  level: 'debug' | 'info' | 'warn' | 'error';
  source: string;
  message: string;
  timestamp: string;
}

// App state
export interface AppState {
  blocks: Block[];
  predicates: Predicate[];
  goal: Predicate[];
  goalName: string;
  goalSatisfied: boolean;
  unsatisfiedGoals: Predicate[];
  plan: PlanAction[];
  currentActionIndex: number;
  cycle: number;
  maxCycles: number;
  executionStatus: ExecutionStatus;
  gripper: GripperState;
  logs: LogEntry[];
  connected: boolean;
  robotConnected: boolean;
  cameraConnected: boolean;
  isConnecting: boolean;
  connectionError: string | null;
}
