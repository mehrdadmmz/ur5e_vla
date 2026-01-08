import { useState } from 'react';
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
  'floating',
  'holding',
  'hand_free',
  'top',
  'nothing_beside',
];

export function StatePanel() {
  const { state } = useRobot();
  const [isResetting, setIsResetting] = useState(false);
  const grouped = groupPredicates(state.predicates);

  const handleReset = async () => {
    setIsResetting(true);
    try {
      const res = await fetch('/api/state/observe', { method: 'POST' });
      const data = await res.json();
      if (!data.success) {
        console.error('Reset observation failed:', data.message);
      }
    } catch (err) {
      console.error('Reset observation error:', err);
    } finally {
      setIsResetting(false);
    }
  };

  return (
    <div className="bg-white rounded-lg p-4 h-full flex flex-col border border-gray-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <h2 className="text-lg font-semibold text-gray-800">World State</h2>
        <button
          onClick={handleReset}
          disabled={isResetting || !state.robotConnected}
          className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
            isResetting || !state.robotConnected
              ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
              : 'bg-blue-500 text-white hover:bg-blue-600'
          }`}
        >
          {isResetting ? 'Observing...' : 'Reset'}
        </button>
      </div>

      {/* Scrollable content area */}
      <div className="flex-1 overflow-y-auto min-h-0">
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
      <div>
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
      </div>{/* End scrollable content area */}
    </div>
  );
}
