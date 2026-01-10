import { ActionIcon, Badge, Group, Text, Button } from "@mantine/core";
import {
  IconLayoutSidebarLeftCollapse,
  IconLayoutSidebarRightCollapse,
  IconPlugConnected,
  IconPlugConnectedX,
  IconRefresh,
} from "@tabler/icons-react";

type ConnectionState = "connected" | "disconnected" | "checking";

type StatusBarProps = {
  sidebarOpened: boolean;
  onToggleSidebar: () => void;
  connectionState: ConnectionState;
};

const statusColorMap: Record<ConnectionState, string> = {
  connected: "green",
  disconnected: "red",
  checking: "yellow",
};

function StatusBar({ sidebarOpened, onToggleSidebar, connectionState }: StatusBarProps) {
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
