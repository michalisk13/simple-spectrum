import { ActionIcon, Badge, Group, Text, Button } from "@mantine/core";
import {
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarRightCollapse,
  IconPlugConnected,
  IconPlugConnectedX,
  IconRefresh,
} from "@tabler/icons-react";

// UI-friendly connection state values used by the status badge.
type ConnectionState = "connected" | "disconnected" | "checking";

type StatusBarProps = {
  sidebarOpened: boolean;
  onToggleSidebar: () => void;
  connectionState: ConnectionState;
  onRefresh: () => void;
};

// Mantine badge colors mapped to connection status.
const statusColorMap: Record<ConnectionState, string> = {
  connected: "green",
  disconnected: "red",
  checking: "yellow",
};

function StatusBar({
  sidebarOpened,
  onToggleSidebar,
  connectionState,
  onRefresh,
}: StatusBarProps) {
  const SidebarIcon = sidebarOpened
    ? IconLayoutSidebarLeftCollapse
    : IconLayoutSidebarRightCollapse;

  return (
    <Group justify="space-between" align="center" className="status-bar">
      <Group gap="md">
        <ActionIcon
          variant="subtle"
          color="gray"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
        >
          <SidebarIcon size={20} />
        </ActionIcon>
        <Text fw={600}>Pluto Spectrum Analyzer</Text>
        {/* Connection badge derived from the latest API status. */}
        <Badge variant="light" color={statusColorMap[connectionState]}>
          {connectionState}
        </Badge>
        <Text size="sm" c="dimmed">
          API: /api/status
        </Text>
      </Group>

      <Group gap="sm">
        <Button
          size="xs"
          variant="light"
          leftSection={<IconRefresh size={16} />}
          onClick={onRefresh}
        >
          Refresh
        </Button>
        <Button
          size="xs"
          variant="filled"
          leftSection={
            connectionState === "connected" ? (
              <IconPlugConnectedX size={16} />
            ) : (
              <IconPlugConnected size={16} />
            )
          }
        >
          {connectionState === "connected" ? "Disconnect" : "Connect"}
        </Button>
      </Group>
    </Group>
  );
}

export default StatusBar;
