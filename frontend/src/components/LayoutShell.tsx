import { AppShell, Box } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { ApiClient } from "../api/client";
import type { EngineStatus } from "../api/types";
import { useWebSocket } from "../hooks/useWebSocket";
import LeftSidebar from "./LeftSidebar";
import SpectrumPanel from "./SpectrumPanel";
import SpectrogramPanel from "./SpectrogramPanel";
import StatusBar from "./StatusBar";

// Local connection state used for the status badge.
type ConnectionState = "connected" | "disconnected" | "checking";

function LayoutShell() {
  const [navbarOpened, { toggle: toggleNavbar }] = useDisclosure(true);
  // Track the latest API status payload for the UI.
  const [status, setStatus] = useState<EngineStatus | null>(null);
  // Track whether we're currently polling the API status endpoint.
  const [isChecking, setIsChecking] = useState(true);
  // Track if a connect/disconnect action is in progress.
  const [isConnecting, setIsConnecting] = useState(false);

  const { statusFrame, latestSpectrumFrameRef, latestSpectrogramFrameRef } =
    useWebSocket();

  // Create a single API client instance for this layout.
  const apiClient = useMemo(() => new ApiClient(), []);

  // Fetch the latest status from the backend.
  const fetchStatus = useCallback(async () => {
    setIsChecking(true);
    const response = await apiClient.getStatus();
    if (response) {
      setStatus(response.status);
    }
    setIsChecking(false);
  }, [apiClient]);

  const handleConnectToggle = useCallback(async () => {
    if (isConnecting) {
      return;
    }
    setIsConnecting(true);
    const response = status?.connected
      ? await apiClient.disconnectSdr()
      : await apiClient.connectSdr();
    if (response?.status) {
      setStatus(response.status);
    }
    setIsConnecting(false);
  }, [apiClient, isConnecting, status]);

  // Load the initial status snapshot when the layout mounts.
  useEffect(() => {
    void fetchStatus();
  }, [fetchStatus]);

  useEffect(() => {
    if (!statusFrame) {
      return;
    }
    setStatus(statusFrame);
    setIsChecking(false);
  }, [statusFrame]);

  // Map status + loading to a friendly UI badge state.
  const connectionState: ConnectionState = isChecking
    ? "checking"
    : status?.connected
      ? "connected"
      : "disconnected";

  return (
    <AppShell
      header={{ height: 64 }}
      navbar={{
        width: 300,
        breakpoint: "sm",
        collapsed: { desktop: !navbarOpened, mobile: !navbarOpened },
      }}
      padding="md"
      transitionDuration={200}
      transitionTimingFunction="ease"
      styles={{
        main: {
          background: "var(--app-background)",
        },
        navbar: {
          background: "var(--panel-background)",
          borderRight: "1px solid var(--panel-border)",
        },
        header: {
          background: "var(--panel-background)",
          borderBottom: "1px solid var(--panel-border)",
        },
      }}
    >
      <AppShell.Header>
        <StatusBar
          sidebarOpened={navbarOpened}
          onToggleSidebar={toggleNavbar}
          connectionState={connectionState}
          onRefresh={fetchStatus}
          onToggleConnection={handleConnectToggle}
          isRefreshing={isChecking}
          isConnecting={isConnecting}
        />
      </AppShell.Header>
      <AppShell.Navbar p="md">
        <LeftSidebar connectionState={connectionState} />
      </AppShell.Navbar>
      <AppShell.Main>
        <Box className="main-panel">
          <PanelGroup direction="vertical">
            <Panel defaultSize={60} minSize={40}>
              <Box className="panel-fill">
                <SpectrumPanel
                  statusFrame={statusFrame}
                  spectrumFrameRef={latestSpectrumFrameRef}
                />
              </Box>
            </Panel>
            <PanelResizeHandle className="resize-handle" />
            <Panel defaultSize={40} minSize={30}>
              <Box className="panel-fill">
                <SpectrogramPanel
                  statusFrame={statusFrame}
                  spectrogramFrameRef={latestSpectrogramFrameRef}
                />
              </Box>
            </Panel>
          </PanelGroup>
        </Box>
      </AppShell.Main>
    </AppShell>
  );
}

export default LayoutShell;
