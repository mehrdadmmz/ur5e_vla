import { RobotProvider } from './context/RobotContext';
import { WebSocketProvider } from './context/WebSocketContext';
import { Header } from './components/layout/Header';
import { Scene3D } from './components/visualization/Scene3D';
import { StatePanel } from './components/state/StatePanel';
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
            <div className="grid grid-cols-12 gap-4" style={{ height: 'calc(100vh - 120px)' }}>
              {/* 3D Visualization - Left side */}
              <div className="col-span-8 row-span-2">
                <Scene3D />
              </div>

              {/* State Panel - Top right */}
              <div className="col-span-4 overflow-hidden">
                <StatePanel />
              </div>

              {/* Plan Panel - Middle right */}
              <div className="col-span-4 overflow-hidden">
                <PlanPanel />
              </div>

              {/* Control Panel - Bottom left */}
              <div className="col-span-4">
                <ControlPanel />
              </div>

              {/* Voice Panel - Bottom middle */}
              <div className="col-span-4">
                <VoicePanel />
              </div>

              {/* Log Panel - Bottom right */}
              <div className="col-span-4">
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
