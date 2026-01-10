import { Group, Paper, Text } from "@mantine/core";
import { IconWaveSine } from "@tabler/icons-react";

function SpectrumPanel() {
  return (
    <Paper className="panel-surface" radius="lg" p="md">
      <Group gap="xs" mb="sm">
        <IconWaveSine size={18} />
        <Text fw={600}>Spectrum</Text>
        <Text size="sm" c="dimmed">
          Live FFT trace
        </Text>
      </Group>
      <div className="panel-placeholder">Spectrum visualization placeholder</div>
    </Paper>
  );
}

export default SpectrumPanel;
