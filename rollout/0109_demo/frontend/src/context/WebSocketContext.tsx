import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useCallback,
  ReactNode,
} from 'react';
import { useRobot } from './RobotContext';
import { WSMessage, LogEntry } from '../types';

interface WebSocketContextValue {
  sendCommand: (command: string, params?: Record<string, unknown>) => void;
  isConnected: boolean;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
  url?: string;
}

export function WebSocketProvider({
  children,
  url,
}: WebSocketProviderProps) {
  // Use relative URL to leverage Vite's proxy in dev mode
  // or same-origin in production
  const wsUrl = url || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
  const { dispatch } = useRobot();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();
  const isConnectedRef = useRef(false);

  const connect = useCallback(() => {
    console.log('[WebSocket] Connecting to', wsUrl);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WebSocket] Connected');
      isConnectedRef.current = true;
      dispatch({ type: 'SET_CONNECTED', connected: true });
      dispatch({
        type: 'ADD_LOG',
        log: {
          level: 'info',
          source: 'system',
          message: 'Connected to server',
          timestamp: new Date().toISOString(),
        },
      });
    };

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'state_update':
            if ('data' in message && message.data) {
              dispatch({
                type: 'UPDATE_STATE',
                data: message.data as Parameters<typeof dispatch>[0] extends { type: 'UPDATE_STATE' } ? Parameters<typeof dispatch>[0]['data'] : never,
              });
            }
            break;

          case 'log':
            if ('data' in message && message.data) {
              dispatch({ type: 'ADD_LOG', log: message.data as LogEntry });
            }
            break;

          case 'logs':
            // Handle bulk logs (on initial connection)
            if ('data' in message && Array.isArray(message.data)) {
              message.data.forEach((log: LogEntry) => {
                dispatch({ type: 'ADD_LOG', log });
              });
            }
            break;

          case 'command_response':
            console.log('[WebSocket] Command response:', message);
            break;

          default:
            console.log('[WebSocket] Unknown message type:', message.type);
        }
      } catch (e) {
        console.error('[WebSocket] Failed to parse message:', e);
      }
    };

    ws.onclose = () => {
      console.log('[WebSocket] Disconnected');
      isConnectedRef.current = false;
      dispatch({ type: 'SET_CONNECTED', connected: false });
      dispatch({
        type: 'ADD_LOG',
        log: {
          level: 'warn',
          source: 'system',
          message: 'Disconnected from server, reconnecting...',
          timestamp: new Date().toISOString(),
        },
      });

      // Reconnect after delay
      reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
    };

    ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };

    wsRef.current = ws;
  }, [wsUrl, dispatch]);

  useEffect(() => {
    connect();

    return () => {
      clearTimeout(reconnectTimeoutRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendCommand = useCallback(
    (command: string, params?: Record<string, unknown>) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const message = {
          type: 'command',
          command,
          params: params || {},
        };
        wsRef.current.send(JSON.stringify(message));
        console.log('[WebSocket] Sent command:', command, params);
      } else {
        console.warn('[WebSocket] Cannot send, not connected');
      }
    },
    []
  );

  return (
    <WebSocketContext.Provider
      value={{
        sendCommand,
        isConnected: isConnectedRef.current,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within WebSocketProvider');
  }
  return context;
}
