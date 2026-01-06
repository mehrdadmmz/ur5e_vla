import { useRobot } from '../../context/RobotContext';
import { PlanAction } from '../../types';

function getStatusColor(status: PlanAction['status']): string {
  switch (status) {
    case 'executing':
      return 'bg-blue-500 text-white';
    case 'completed':
      return 'bg-green-500 text-white';
    case 'failed':
      return 'bg-red-500 text-white';
    default:
      return 'bg-gray-200 text-gray-600';
  }
}

function getStatusIcon(status: PlanAction['status']): string {
  switch (status) {
    case 'executing':
      return '▶';
    case 'completed':
      return '✓';
    case 'failed':
      return '✗';
    default:
      return '○';
  }
}

export function PlanPanel() {
  const { state } = useRobot();
  const plan = state.plan ?? [];

  return (
    <div className="bg-white rounded-lg p-4 h-full flex flex-col border border-gray-200 shadow-sm">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-lg font-semibold text-gray-800">Plan</h2>
        {plan.length > 0 && (
          <span className="text-sm text-gray-500">
            {plan.filter((a) => a.status === 'completed').length}/{plan.length} done
          </span>
        )}
      </div>

      {/* Current action highlight */}
      {state.currentActionIndex >= 0 && plan[state.currentActionIndex] && (
        <div className="mb-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="text-xs text-blue-600 mb-1">Current Action</div>
          <div className="text-gray-800 font-mono">
            {plan[state.currentActionIndex].name}(
            {plan[state.currentActionIndex].args.join(', ')})
          </div>
        </div>
      )}

      {/* Plan steps */}
      <div className="flex-1 overflow-y-auto">
        {plan.length === 0 ? (
          <div className="text-gray-400 text-sm text-center py-8">No plan available</div>
        ) : (
          <div className="space-y-1">
            {plan.map((action, index) => (
              <div
                key={index}
                className={`flex items-center gap-2 p-2 rounded ${
                  index === state.currentActionIndex
                    ? 'bg-blue-50 border border-blue-300'
                    : 'hover:bg-gray-50'
                }`}
              >
                <span
                  className={`w-6 h-6 flex items-center justify-center rounded text-xs ${getStatusColor(
                    action.status
                  )}`}
                >
                  {getStatusIcon(action.status)}
                </span>
                <span className="text-sm font-mono text-gray-700">
                  {action.name}({action.args.join(', ')})
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Plan summary */}
      {plan.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
          {plan.map((a) => `${a.name}(${a.args.join(',')})`).join(' → ')}
        </div>
      )}
    </div>
  );
}
