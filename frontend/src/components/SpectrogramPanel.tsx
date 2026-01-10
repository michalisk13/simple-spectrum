import { Badge, Group, Paper, Text } from "@mantine/core";
import { IconChartHistogram } from "@tabler/icons-react";
import { useCallback, useRef, useState } from "react";
import SpectrogramCanvas from "./plots/SpectrogramCanvas";
import { useAnimationFrame } from "../hooks/useAnimationFrame";
import { generateSpectrogramRow } from "../utils/mockData";

const ROW_POINTS = 512;

function SpectrogramPanel() {
  // Track mock phase so the waterfall bands drift over time.
  const phaseRef = useRef(0);
  // Store the latest row so the canvas updates as new data arrives.
  const [row, setRow] = useState(() =>
    generateSpectrogramRow(ROW_POINTS, phaseRef.current),
  );

  const updateRow = useCallback((deltaMs: number) => {
    phaseRef.current += deltaMs * 0.0016;
    setRow(generateSpectrogramRow(ROW_POINTS, phaseRef.current));
  }, []);

  // Drive the mock spectrogram at 20 fps to validate the buffer pipeline.
  useAnimationFrame(updateRow, 20);

  return (
    <Paper className="panel-surface" radius="lg" p="md">
      <Group justify="space-between" align="center" className="panel-header">
        <Group gap="xs">
          <IconChartHistogram size={18} />
          <Text fw={600}>Spectrogram</Text>
          <Text size="sm" c="dimmed">
            Time-frequency view
          </Text>
        </Group>
        <Group gap="xs">
          <Badge variant="light">Mode: PSD</Badge>
          <Badge variant="light">15 slices/s</Badge>
        </Group>
      </Group>
      <Group gap="lg" className="panel-meta">
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Time span
          </Text>
          <Text size="sm" fw={500}>
            20 s
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Colormap
          </Text>
          <Text size="sm" fw={500}>
            Viridis
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Range
          </Text>
          <Text size="sm" fw={500}>
            -120 dB to 0 dB
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Auto range
          </Text>
          <Text size="sm" fw={500}>
            ±2σ
          </Text>
        </div>
      </Group>
      <div className="plot-container">
        <SpectrogramCanvas row={row} />
      </div>
    </Paper>
  );
}

export default SpectrogramPanel;
