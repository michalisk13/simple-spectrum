import { AppShell, Box } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import LeftSidebar from "./LeftSidebar";
import SpectrumPanel from "./SpectrumPanel";
import SpectrogramPanel from "./SpectrogramPanel";
import StatusBar from "./StatusBar";

function LayoutShell() {
  const [navbarOpened, { toggle: toggleNavbar }] = useDisclosure(true);

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
          connectionState="disconnected"
        />
      </AppShell.Header>
      <AppShell.Navbar p="md">
        <LeftSidebar />
      </AppShell.Navbar>
      <AppShell.Main>
        <Box className="main-panel">
          <PanelGroup direction="vertical">
            <Panel defaultSize={60} minSize={40}>
              <SpectrumPanel />
            </Panel>
            <PanelResizeHandle className="resize-handle" />
            <Panel defaultSize={40} minSize={30}>
              <SpectrogramPanel />
            </Panel>
          </PanelGroup>
        </Box>
      </AppShell.Main>
    </AppShell>
  );
}

export default LayoutShell;
