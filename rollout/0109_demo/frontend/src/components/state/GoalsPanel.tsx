import { useRobot } from '../../context/RobotContext';

export function GoalsPanel() {
  const { state } = useRobot();

  if (!state.goal || state.goal.length === 0) {
    return (
      <div className="bg-white rounded-lg p-4 h-full flex flex-col border border-gray-200 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-800 mb-3">Goals</h2>
        <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
          No goal set
        </div>
      </div>
    );
  }

  const satisfiedCount = state.goal.length - state.unsatisfiedGoals.length;

  return (
    <div className="bg-white rounded-lg p-4 h-full flex flex-col border border-gray-200 shadow-sm">
      {/* Header with goal name and progress */}
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-gray-800">Goals</h2>
          <span className="text-sm text-gray-500">
            {state.goalName || 'Custom'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">
            {satisfiedCount}/{state.goal.length} satisfied
          </span>
          {state.goalSatisfied ? (
            <span className="px-2 py-1 bg-green-500 text-white text-xs font-medium rounded">
              Complete
            </span>
          ) : (
            <span className="px-2 py-1 bg-yellow-500 text-white text-xs font-medium rounded">
              In Progress
            </span>
          )}
        </div>
      </div>

      {/* Goal predicates in a horizontal grid */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="flex flex-wrap gap-2">
          {state.goal.map((pred, idx) => {
            const isSatisfied = !state.unsatisfiedGoals.some(
              (u) => u.name === pred.name && u.args.join(',') === pred.args.join(',')
            );
            return (
              <div
                key={idx}
                className={`px-3 py-2 rounded text-sm ${
                  isSatisfied
                    ? 'bg-green-100 text-green-700 border border-green-200'
                    : 'bg-yellow-100 text-yellow-700 border border-yellow-200'
                }`}
              >
                <span className="font-medium">{pred.name}</span>
                <span className="text-gray-600">({pred.args.join(', ')})</span>
                {isSatisfied && <span className="ml-2 text-green-600">&#10003;</span>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
