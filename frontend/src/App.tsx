import { useEffect, useState } from "react";
import "./App.css";

type StatusResponse = {
  status: {
    connected: boolean;
  };
};

const statusText = (connected: boolean | null) => {
  if (connected === null) {
    return "checking";
  }
  return connected ? "connected" : "disconnected";
};

function App() {
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    let isMounted = true;
    const fetchStatus = async () => {
      try {
        const response = await fetch("/api/status");
        if (!response.ok) {
          throw new Error(`Status request failed: ${response.status}`);
        }
        const payload = (await response.json()) as StatusResponse;
        if (isMounted) {
          setConnected(Boolean(payload.status?.connected));
        }
      } catch (error) {
        if (isMounted) {
          setConnected(false);
        }
        console.error(error);
      }
    };

    fetchStatus();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <header className="panel-header">
          <h1>Settings</h1>
        </header>
        <section className="panel-body">
          <div className="placeholder">Preset controls</div>
          <div className="placeholder">Frequency controls</div>
          <div className="placeholder">Gain &amp; display</div>
        </section>
      </aside>
      <main className="main-panel">
        <section className="panel spectrum-panel">
          <header className="panel-header">
            <h2>Spectrum Plot</h2>
          </header>
          <div className="panel-body placeholder">Spectrum visualization placeholder</div>
        </section>
        <section className="panel spectrogram-panel">
          <header className="panel-header">
            <h2>Spectrogram</h2>
          </header>
          <div className="panel-body placeholder">Spectrogram placeholder</div>
        </section>
      </main>
      <footer className="status-bar">
        <div className="status-indicator">
          <span className={`status-dot ${statusText(connected)}`} aria-hidden="true" />
          <span className="status-label">
            {statusText(connected)}
          </span>
        </div>
        <span className="status-details">API: /api/status</span>
      </footer>
    </div>
  );
}

export default App;
