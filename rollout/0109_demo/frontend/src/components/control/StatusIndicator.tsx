import { ExecutionStatus } from '../../types';

interface StatusIndicatorProps {
  status: ExecutionStatus;
}

const STATUS_CONFIG: Record<
  ExecutionStatus,
  { label: string; color: string; bgColor: string; animate?: boolean }
> = {
  idle: { label: 'Idle', color: 'text-gray-500', bgColor: 'bg-gray-400' },
  observing: {
    label: 'Observing',
    color: 'text-cyan-600',
    bgColor: 'bg-cyan-500',
    animate: true,
  },
  planning: {
    label: 'Planning',
    color: 'text-purple-600',
    bgColor: 'bg-purple-500',
    animate: true,
  },
  executing: {
    label: 'Executing',
    color: 'text-blue-600',
    bgColor: 'bg-blue-500',
    animate: true,
  },
  paused: { label: 'Paused', color: 'text-yellow-600', bgColor: 'bg-yellow-500' },
  error: { label: 'Error', color: 'text-red-600', bgColor: 'bg-red-500' },
  completed: { label: 'Completed', color: 'text-green-600', bgColor: 'bg-green-500' },
};

export function StatusIndicator({ status }: StatusIndicatorProps) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.idle;

  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-3 h-3 rounded-full ${config.bgColor} ${
          config.animate ? 'animate-pulse' : ''
        }`}
      />
      <span className={`text-sm font-medium ${config.color}`}>{config.label}</span>
    </div>
  );
}
