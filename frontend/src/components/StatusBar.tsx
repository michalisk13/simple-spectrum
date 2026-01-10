import { ActionIcon, Badge, Button, Group, Stack, Text } from "@mantine/core";
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
  onToggleConnection: () => void;
  isRefreshing: boolean;
  isConnecting: boolean;
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
  onToggleConnection,
  isRefreshing,
  isConnecting,
}: StatusBarProps) {
  const SidebarIcon = sidebarOpened
    ? IconLayoutSidebarLeftCollapse
    : IconLayoutSidebarRightCollapse;

  return (
    <Group justify="space-between" align="center" className="status-bar">
      <Group gap="md" className="status-left">
        <ActionIcon
          variant="subtle"
          color="gray"
          onClick={onToggleSidebar}
          aria-label="Toggle sidebar"
        >
          <SidebarIcon size={20} />
        </ActionIcon>
        <Group gap="xs">
          <Text fw={600}>Pluto Spectrum Analyzer</Text>
          <Badge variant="light" color={statusColorMap[connectionState]}>
            {connectionState}
          </Badge>
        </Group>
        <Group gap="md" className="status-meta">
          <Text size="sm" c="dimmed">
            SDR: Pluto (ip:192.168.2.1)
          </Text>
          <Text size="sm" c="dimmed">
            Center: 2.437 GHz
          </Text>
          <Text size="sm" c="dimmed">
            Span: 20 MHz
          </Text>
        </Group>
      </Group>

      <Group gap="lg" className="status-right">
        <Stack gap={2} className="status-messages">
          <Text size="xs" c="dimmed">
            Instrument: Ready
          </Text>
          <Text size="xs" c="dimmed">
            Status: Idle
          </Text>
          <Text size="xs" c="yellow">
            Warnings: None
          </Text>
        </Stack>
        <Group gap="sm">
          <Button
            size="xs"
            variant="light"
            leftSection={<IconRefresh size={16} />}
            onClick={onRefresh}
            loading={isRefreshing}
            disabled={isConnecting}
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
            onClick={onToggleConnection}
            loading={isConnecting}
            disabled={isRefreshing}
          >
            {connectionState === "connected" ? "Disconnect" : "Connect"}
          </Button>
        </Group>
      </Group>
    </Group>
  );
}

export default StatusBar;
