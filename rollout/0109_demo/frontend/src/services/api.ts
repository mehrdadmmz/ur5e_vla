/**
 * API service for communicating with the backend.
 */

const API_BASE = '/api';

export interface ConnectResponse {
  success: boolean;
  message: string;
  robot_connected: boolean;
  camera_connected: boolean;
}

export interface SystemStatus {
  initialized: boolean;
  robot_connected: boolean;
  camera_connected: boolean;
  execution_status: string;
}

/**
 * Connect to the robot and camera.
 */
export async function connectToRobot(): Promise<ConnectResponse> {
  const response = await fetch(`${API_BASE}/system/connect`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  return response.json();
}

/**
 * Get current system status.
 */
export async function getSystemStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE}/system/status`);

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  return response.json();
}
