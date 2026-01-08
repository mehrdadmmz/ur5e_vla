import { RobotProvider } from './context/RobotContext';
import { WebSocketProvider } from './context/WebSocketContext';
import { Header } from './components/layout/Header';
import { Scene3D } from './components/visualization/Scene3D';
import { StatePanel } from './components/state/StatePanel';
import { GoalsPanel } from './components/state/GoalsPanel';
import { PlanPanel } from './components/plan/PlanPanel';
import { ControlPanel } from './components/control/ControlPanel';
import { LogPanel } from './components/logs/LogPanel';
import { VoicePanel } from './components/voice/VoicePanel';

function App() {
  return (
    <RobotProvider>
      <WebSocketProvider>
        <div className="min-h-screen bg-gray-100 text-gray-900">
          <Header />

          <main className="container mx-auto px-4 py-4">
            <div className="grid grid-cols-12 gap-3" style={{ height: 'calc(100vh - 120px)' }}>
              {/* Row 1: 3D (5) + State (7) */}
              <div className="col-span-5">
                <Scene3D />
              </div>
              <div className="col-span-7 overflow-hidden">
                <StatePanel />
              </div>

              {/* Row 2: Control (3) + Voice (2) + Plan (4) + Goals (3) */}
              <div className="col-span-3">
                <ControlPanel />
              </div>
              <div className="col-span-2">
                <VoicePanel />
              </div>
              <div className="col-span-4 overflow-hidden">
                <PlanPanel />
              </div>
              <div className="col-span-3 overflow-hidden">
                <GoalsPanel />
              </div>

              {/* Row 3: Logs (full width, fixed 200px height) */}
              <div className="col-span-12 h-[200px]">
                <LogPanel />
              </div>
            </div>
          </main>
        </div>
      </WebSocketProvider>
    </RobotProvider>
  );
}

export default App;
