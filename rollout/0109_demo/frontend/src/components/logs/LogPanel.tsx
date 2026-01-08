import { useRef, useEffect, useState } from 'react';
import { useRobot } from '../../context/RobotContext';
import { LogEntry } from '../../types';

const LEVEL_COLORS: Record<string, string> = {
  debug: 'text-gray-500',
  info: 'text-blue-600',
  warn: 'text-yellow-600',
  error: 'text-red-600',
};

const LEVEL_BG: Record<string, string> = {
  debug: 'bg-gray-50',
  info: 'bg-gray-50',
  warn: 'bg-yellow-50',
  error: 'bg-red-50',
};

function LogLine({ log }: { log: LogEntry }) {
  const time = log.timestamp.split('T')[1]?.substring(0, 8) || log.timestamp;

  return (
    <div className={`flex gap-2 py-1 px-2 text-xs font-mono ${LEVEL_BG[log.level]}`}>
      <span className="text-gray-500 w-16 flex-shrink-0">{time}</span>
      <span className={`w-12 flex-shrink-0 ${LEVEL_COLORS[log.level]}`}>
        [{log.level.toUpperCase()}]
      </span>
      <span className="text-gray-500 w-16 flex-shrink-0">{log.source}</span>
      <span className="text-gray-700 flex-1">{log.message}</span>
    </div>
  );
}

export function LogPanel() {
  const { state, dispatch } = useRobot();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filter, setFilter] = useState<string | null>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [state.logs, autoScroll]);

  // Filter logs
  const filteredLogs = filter
    ? state.logs.filter((log) => log.level === filter)
    : state.logs;

  const handleClear = () => {
    dispatch({ type: 'CLEAR_LOGS' });
  };

  return (
    <div className="bg-white rounded-lg p-3 h-full flex flex-col border border-gray-200 shadow-sm min-h-0">
      <div className="flex justify-between items-center mb-2 flex-shrink-0">
        <h2 className="text-base font-semibold text-gray-800">Logs</h2>
        <div className="flex items-center gap-2">
          {/* Filter buttons */}
          <div className="flex gap-1">
            {['info', 'warn', 'error'].map((level) => (
              <button
                key={level}
                onClick={() => setFilter(filter === level ? null : level)}
                className={`px-2 py-0.5 text-xs rounded ${
                  filter === level
                    ? LEVEL_COLORS[level] + ' bg-gray-200'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {level}
              </button>
            ))}
          </div>

          {/* Auto-scroll toggle */}
          <button
            onClick={() => setAutoScroll(!autoScroll)}
            className={`px-2 py-0.5 text-xs rounded ${
              autoScroll ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-600'
            }`}
            title="Auto-scroll"
          >
            ↓
          </button>

          {/* Clear button */}
          <button
            onClick={handleClear}
            className="px-2 py-0.5 text-xs bg-gray-200 hover:bg-gray-300 text-gray-600 rounded"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Log content */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto bg-gray-50 rounded border border-gray-200 min-h-0"
      >
        {filteredLogs.length === 0 ? (
          <div className="text-gray-500 text-sm text-center py-8">No logs</div>
        ) : (
          filteredLogs.map((log, index) => <LogLine key={index} log={log} />)
        )}
      </div>

      {/* Log count */}
      <div className="mt-1 text-xs text-gray-500 text-right flex-shrink-0">
        {filteredLogs.length} / {state.logs.length} entries
      </div>
    </div>
  );
}
