import { Badge, Box, Divider, Group, Stack, Text } from "@mantine/core";
import {
  IconAdjustments,
  IconRadio,
  IconSliders,
  IconStar,
} from "@tabler/icons-react";

const sectionHeaderProps = {
  size: "xs",
  fw: 600,
  tt: "uppercase",
  c: "dimmed",
  className: "sidebar-section-title",
} as const;

function LeftSidebar() {
  return (
    <Stack gap="lg" className="sidebar">
      <Group justify="space-between" align="center">
        <Group gap="sm">
          <IconRadio size={20} />
          <Text fw={600} size="lg">
            Controls
          </Text>
        </Group>
        <Badge variant="light" color="red">
          Offline
        </Badge>
      </Group>

      <Box>
        <Group gap="xs" mb="sm">
          <IconStar size={16} />
          <Text {...sectionHeaderProps}>Presets</Text>
        </Group>
        <Stack gap="xs">
          <Box className="sidebar-card">Fast View</Box>
          <Box className="sidebar-card">Wide Scan</Box>
          <Box className="sidebar-card">Measure</Box>
        </Stack>
      </Box>

      <Divider />

      <Box>
        <Group gap="xs" mb="sm">
          <IconSliders size={16} />
          <Text {...sectionHeaderProps}>Frequency</Text>
        </Group>
        <Stack gap="xs">
          <Box className="sidebar-card">Center &amp; Span</Box>
          <Box className="sidebar-card">RBW / VBW</Box>
          <Box className="sidebar-card">Markers</Box>
        </Stack>
      </Box>

      <Divider />

      <Box>
        <Group gap="xs" mb="sm">
          <IconAdjustments size={16} />
          <Text {...sectionHeaderProps}>Gain &amp; Display</Text>
        </Group>
        <Stack gap="xs">
          <Box className="sidebar-card">Gain &amp; Atten</Box>
          <Box className="sidebar-card">Scale &amp; Ref</Box>
          <Box className="sidebar-card">Spectrogram</Box>
        </Stack>
      </Box>
    </Stack>
  );
}

export default LeftSidebar;
