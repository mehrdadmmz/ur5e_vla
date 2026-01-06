import { useRobot } from '../../context/RobotContext';
import { StatusIndicator } from '../control/StatusIndicator';
import { connectToRobot } from '../../services/api';

export function Header() {
  const { state, dispatch } = useRobot();

  const isConnected = state.robotConnected && state.cameraConnected;
  const hasError = state.connectionError !== null;

  const handleConnect = async () => {
    if (state.isConnecting || isConnected) return;

    dispatch({ type: 'SET_CONNECTING', isConnecting: true });
    dispatch({ type: 'SET_CONNECTION_ERROR', error: null });

    try {
      const result = await connectToRobot();

      if (result.success) {
        dispatch({
          type: 'SET_ROBOT_CONNECTED',
          robotConnected: result.robot_connected,
          cameraConnected: result.camera_connected,
        });
      } else {
        dispatch({ type: 'SET_CONNECTION_ERROR', error: result.message });
      }
    } catch (error) {
      dispatch({
        type: 'SET_CONNECTION_ERROR',
        error: error instanceof Error ? error.message : 'Connection failed',
      });
    }
  };

  // Determine button state and styling
  const getButtonConfig = () => {
    if (state.isConnecting) {
      return {
        text: 'Connecting...',
        className: 'bg-yellow-500 text-white cursor-wait',
        disabled: true,
      };
    }
    if (isConnected) {
      return {
        text: 'Connected',
        className: 'bg-green-500 text-white cursor-default',
        disabled: true,
      };
    }
    if (hasError) {
      return {
        text: 'Retry',
        className: 'bg-red-500 hover:bg-red-600 text-white cursor-pointer',
        disabled: false,
      };
    }
    return {
      text: 'Connect',
      className: 'bg-gray-500 hover:bg-gray-600 text-white cursor-pointer',
      disabled: false,
    };
  };

  const buttonConfig = getButtonConfig();

  return (
    <header className="bg-white border-b border-gray-200 px-4 py-3 shadow-sm">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-bold text-gray-800">Block World Control</h1>
          {state.goalName && (
            <span className="px-3 py-1 bg-blue-500 text-white text-sm rounded-full">
              Goal: {state.goalName}
            </span>
          )}
        </div>

        <div className="flex items-center gap-4">
          <StatusIndicator status={state.executionStatus} />

          {/* Connection Status with Connect Button */}
          <div className="flex items-center gap-3 px-3 py-1 bg-gray-100 rounded-lg">
            {/* Server Connection */}
            <div className="flex items-center gap-1.5" title="Backend Server">
              <div
                className={`w-2 h-2 rounded-full ${
                  state.connected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-xs text-gray-600">Server</span>
            </div>

            <div className="w-px h-4 bg-gray-300" />

            {/* Robot Connection */}
            <div className="flex items-center gap-1.5" title="Robot (UR5e)">
              <div
                className={`w-2 h-2 rounded-full ${
                  state.robotConnected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-xs text-gray-600">Robot</span>
            </div>

            <div className="w-px h-4 bg-gray-300" />

            {/* Camera Connection */}
            <div className="flex items-center gap-1.5" title="Camera (RealSense)">
              <div
                className={`w-2 h-2 rounded-full ${
                  state.cameraConnected ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-xs text-gray-600">Camera</span>
            </div>

            <div className="w-px h-4 bg-gray-300" />

            {/* Connect Button */}
            <button
              onClick={handleConnect}
              disabled={buttonConfig.disabled}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${buttonConfig.className}`}
              title={state.connectionError || 'Connect to robot and camera'}
            >
              {state.isConnecting && (
                <span className="inline-block w-3 h-3 mr-1 border-2 border-white border-t-transparent rounded-full animate-spin" />
              )}
              {buttonConfig.text}
            </button>
          </div>

          <div className="text-sm text-gray-600">
            Cycle: {state.cycle}/{state.maxCycles}
          </div>
        </div>
      </div>

      {/* Error message banner */}
      {state.connectionError && (
        <div className="container mx-auto mt-2">
          <div className="bg-red-100 border border-red-300 text-red-700 px-3 py-1 rounded text-sm">
            {state.connectionError}
          </div>
        </div>
      )}
    </header>
  );
}
