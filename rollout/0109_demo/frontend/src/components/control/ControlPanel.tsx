import { useState, useEffect } from 'react';
import { useWebSocket } from '../../context/WebSocketContext';
import { useRobot } from '../../context/RobotContext';

interface GoalOption {
  name: string;
  label: string;
  description?: string;
}

export function ControlPanel() {
  const { state } = useRobot();
  const { sendCommand } = useWebSocket();
  const [selectedGoal, setSelectedGoal] = useState('');
  const [availableGoals, setAvailableGoals] = useState<GoalOption[]>([]);
  const [goalsLoading, setGoalsLoading] = useState(true);
  const [isMovingHome, setIsMovingHome] = useState(false);

  // Fetch available goals from backend
  useEffect(() => {
    fetch('/api/state/goals')
      .then((res) => res.json())
      .then((data) => {
        const goals = data.goals.map((g: { name: string; description: string }) => ({
          name: g.name,
          label: g.name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (c: string) => c.toUpperCase()),
          description: g.description,
        }));
        setAvailableGoals(goals);
        if (goals.length > 0 && !selectedGoal) {
          setSelectedGoal(goals[0].name);
        }
        setGoalsLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch goals:', err);
        setGoalsLoading(false);
      });
  }, []);

  const isPaused = state.executionStatus === 'paused';
  const isExecuting = state.executionStatus === 'executing';
  const isIdle = state.executionStatus === 'idle' || state.executionStatus === 'completed';

  const handlePause = () => sendCommand('pause');
  const handleContinue = () => sendCommand('continue');
  const handleQuit = () => sendCommand('quit');

  const handleSetGoal = () => {
    sendCommand('set_goal', { goal_name: selectedGoal });
  };

  const handleMoveHome = async () => {
    setIsMovingHome(true);
    try {
      const res = await fetch('/api/control/home', { method: 'POST' });
      const data = await res.json();
      if (!data.success) {
        console.error('Move home failed:', data.message);
      }
    } catch (err) {
      console.error('Move home error:', err);
    } finally {
      setIsMovingHome(false);
    }
  };

  return (
    <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">Control</h2>

      {/* Execution controls */}
      <div className="flex gap-2 mb-4">
        {isPaused ? (
          <button
            onClick={handleContinue}
            className="flex-1 px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-lg font-medium transition-colors"
          >
            Continue
          </button>
        ) : (
          <button
            onClick={handlePause}
            disabled={!isExecuting}
            className="flex-1 px-4 py-2 bg-yellow-500 hover:bg-yellow-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
          >
            Pause
          </button>
        )}

        <button
          onClick={handleQuit}
          disabled={isIdle}
          className="px-4 py-2 bg-gray-500 hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          Quit
        </button>

        <button
          onClick={handleMoveHome}
          disabled={isExecuting || isMovingHome || !state.robotConnected}
          className="px-4 py-2 bg-indigo-500 hover:bg-indigo-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
        >
          {isMovingHome ? 'Moving...' : 'Home'}
        </button>
      </div>

      {/* Goal selection */}
      <div className="border-t border-gray-200 pt-4">
        <label className="block text-sm text-gray-600 mb-2">Change Goal</label>
        <div className="flex gap-2">
          <select
            value={selectedGoal}
            onChange={(e) => setSelectedGoal(e.target.value)}
            disabled={goalsLoading || availableGoals.length === 0}
            className="flex-1 px-3 py-2 bg-white border border-gray-300 rounded-lg text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          >
            {goalsLoading ? (
              <option>Loading goals...</option>
            ) : availableGoals.length === 0 ? (
              <option>No goals available</option>
            ) : (
              availableGoals.map((goal) => (
                <option key={goal.name} value={goal.name} title={goal.description}>
                  {goal.label}
                </option>
              ))
            )}
          </select>
          <button
            onClick={handleSetGoal}
            disabled={goalsLoading || !selectedGoal}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
          >
            Set
          </button>
        </div>
      </div>

      {/* Gripper status */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Gripper</span>
          <span className={state.gripper.open ? 'text-green-600' : 'text-yellow-600'}>
            {state.gripper.open ? 'Open' : 'Closed'}
            {state.gripper.holding !== null && ` (holding block ${state.gripper.holding})`}
          </span>
        </div>
        <div className="mt-1 text-xs text-gray-500 font-mono">
          Pos: [{state.gripper.position.map((v) => v.toFixed(3)).join(', ')}]
        </div>
      </div>
    </div>
  );
}
