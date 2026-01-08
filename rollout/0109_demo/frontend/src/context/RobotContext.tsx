import { createContext, useContext, useReducer, ReactNode } from 'react';
import {
  AppState,
  Block,
  BlockObservation,
  Predicate,
  PlanAction,
  ExecutionStatus,
  LogEntry,
} from '../types';

// Block colors by state
const STATE_COLORS: Record<string, string> = {
  'on-table': '#4ade80', // Green
  stacked: '#60a5fa', // Blue
  held: '#fbbf24', // Yellow
  moving: '#f87171', // Red
};

function computeBlockState(
  blockId: number,
  predicates: Predicate[],
  holdingBlock: number | null
): string {
  if (holdingBlock === blockId) return 'held';

  const isAbove = predicates.some(
    (p) => p.name === 'above' && p.args[0] === String(blockId)
  );
  if (isAbove) return 'stacked';

  return 'on-table';
}

function convertToBlocks(
  observations: BlockObservation[],
  predicates: Predicate[],
  holdingBlock: number | null
): Block[] {
  return observations.map((obs) => {
    const state = computeBlockState(obs.id, predicates, holdingBlock) as Block['state'];
    return {
      ...obs,
      state,
      color: STATE_COLORS[state] || '#9ca3af',
    };
  });
}

// Initial state
const initialState: AppState = {
  blocks: [],
  predicates: [],
  goal: [],
  goalName: '',
  goalSatisfied: false,
  unsatisfiedGoals: [],
  plan: [],
  currentActionIndex: -1,
  cycle: 0,
  maxCycles: 100,
  executionStatus: 'idle',
  gripper: {
    position: [0, 0.5, 0.6],
    open: true,
    holding: null,
  },
  logs: [],
  connected: false,
  robotConnected: false,
  cameraConnected: false,
  isConnecting: false,
  connectionError: null,
};

// Action types
type Action =
  | {
      type: 'UPDATE_STATE';
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
      };
    }
  | { type: 'ADD_LOG'; log: LogEntry }
  | { type: 'CLEAR_LOGS' }
  | { type: 'SET_CONNECTED'; connected: boolean }
  | { type: 'SET_CONNECTING'; isConnecting: boolean }
  | { type: 'SET_CONNECTION_ERROR'; error: string | null }
  | { type: 'SET_ROBOT_CONNECTED'; robotConnected: boolean; cameraConnected: boolean };

// Reducer
function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'UPDATE_STATE': {
      const { data } = action;
      const blocks = convertToBlocks(
        data.blocks,
        data.predicates,
        data.holding_block
      );
      return {
        ...state,
        blocks,
        predicates: data.predicates,
        goal: data.goal,
        goalName: data.goal_name,
        goalSatisfied: data.goal_satisfied,
        unsatisfiedGoals: data.unsatisfied_goals,
        plan: data.current_plan,
        currentActionIndex: data.current_action_index,
        cycle: data.cycle,
        maxCycles: data.max_cycles,
        executionStatus: data.execution_status,
        gripper: {
          position: data.gripper_position,
          open: data.gripper_open,
          holding: data.holding_block,
        },
        robotConnected: data.robot_connected,
        cameraConnected: data.camera_connected,
      };
    }
    case 'ADD_LOG':
      return {
        ...state,
        logs: [...state.logs.slice(-199), action.log],
      };
    case 'CLEAR_LOGS':
      return { ...state, logs: [] };
    case 'SET_CONNECTED':
      return { ...state, connected: action.connected };
    case 'SET_CONNECTING':
      return { ...state, isConnecting: action.isConnecting };
    case 'SET_CONNECTION_ERROR':
      return { ...state, connectionError: action.error, isConnecting: false };
    case 'SET_ROBOT_CONNECTED':
      return {
        ...state,
        robotConnected: action.robotConnected,
        cameraConnected: action.cameraConnected,
        isConnecting: false,
        connectionError: null,
      };
    default:
      return state;
  }
}

// Context
interface RobotContextValue {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}

const RobotContext = createContext<RobotContextValue | null>(null);

// Provider
export function RobotProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  return (
    <RobotContext.Provider value={{ state, dispatch }}>
      {children}
    </RobotContext.Provider>
  );
}

// Hook
export function useRobot() {
  const context = useContext(RobotContext);
  if (!context) {
    throw new Error('useRobot must be used within RobotProvider');
  }
  return context;
}
