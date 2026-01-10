import { Group, Paper, Text } from "@mantine/core";
import { IconChartHistogram } from "@tabler/icons-react";

function SpectrogramPanel() {
  return (
    <Paper className="panel-surface" radius="lg" p="md">
      <Group gap="xs" mb="sm">
        <IconChartHistogram size={18} />
        <Text fw={600}>Spectrogram</Text>
        <Text size="sm" c="dimmed">
          Time-frequency view
        </Text>
      </Group>
      <div className="panel-placeholder">Spectrogram placeholder</div>
    </Paper>
  );
}

export default SpectrogramPanel;
