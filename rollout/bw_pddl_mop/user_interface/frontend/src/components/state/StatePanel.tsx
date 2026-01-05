import { useRobot } from '../../context/RobotContext';
import { Predicate } from '../../types';

// Group predicates by type
function groupPredicates(predicates: Predicate[]): Record<string, Predicate[]> {
  const groups: Record<string, Predicate[]> = {};
  for (const pred of predicates) {
    if (!groups[pred.name]) {
      groups[pred.name] = [];
    }
    groups[pred.name].push(pred);
  }
  return groups;
}

// Predicate display order
const PREDICATE_ORDER = [
  'on-table',
  'above',
  'beside',
  'above_both',
  'holding',
  'hand_free',
  'top',
  'nothing_beside',
];

export function StatePanel() {
  const { state } = useRobot();
  const grouped = groupPredicates(state.predicates);

  return (
    <div className="bg-white rounded-lg p-4 h-full flex flex-col border border-gray-200 shadow-sm">
      <h2 className="text-lg font-semibold text-gray-800 mb-3">World State</h2>

      {/* Block summary */}
      <div className="mb-3 text-sm text-gray-600">
        {state.blocks.length} blocks detected
        {state.blocks.some((b) => b.stale) && (
          <span className="text-red-600 ml-2">
            ({state.blocks.filter((b) => b.stale).length} stale)
          </span>
        )}
      </div>

      {/* Block positions */}
      <div className="mb-4">
        <h3 className="text-sm font-medium text-gray-500 mb-2">Positions</h3>
        <div className="flex flex-wrap gap-2">
          {state.blocks.map((block) => (
            <div
              key={block.id}
              className={`px-2 py-1 rounded text-xs font-mono ${
                block.stale
                  ? 'bg-red-100 text-red-700 border border-red-200'
                  : block.confident
                  ? 'bg-gray-100 text-gray-700 border border-gray-200'
                  : 'bg-yellow-100 text-yellow-700 border border-yellow-200'
              }`}
            >
              {block.id}: [{block.position[0].toFixed(2)}, {block.position[1].toFixed(2)},{' '}
              {block.position[2].toFixed(2)}]
            </div>
          ))}
        </div>
      </div>

      {/* Predicates */}
      <div className="flex-1 overflow-y-auto">
        <h3 className="text-sm font-medium text-gray-500 mb-2">Predicates</h3>
        <div className="space-y-2">
          {PREDICATE_ORDER.map((type) =>
            grouped[type] ? (
              <div key={type} className="text-sm">
                <span className="text-blue-600 font-medium">{type}:</span>{' '}
                <span className="text-gray-700">
                  {grouped[type].map((p) => p.args.join(',')).join(' | ')}
                </span>
              </div>
            ) : null
          )}
        </div>
      </div>

      {/* Goal status */}
      {state.goalName && (
        <div className="mt-4 pt-3 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500">Goal Status</span>
            {state.goalSatisfied ? (
              <span className="px-2 py-1 bg-green-500 text-white text-xs rounded">
                Satisfied
              </span>
            ) : (
              <span className="px-2 py-1 bg-yellow-500 text-white text-xs rounded">
                In Progress
              </span>
            )}
          </div>
          {!state.goalSatisfied && state.unsatisfiedGoals.length > 0 && (
            <div className="mt-2 text-xs text-gray-500">
              Missing:{' '}
              {state.unsatisfiedGoals.map((g) => `${g.name}(${g.args.join(',')})`).join(', ')}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
