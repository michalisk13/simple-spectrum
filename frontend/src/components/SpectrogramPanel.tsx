import { Group, Paper, Text } from "@mantine/core";
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
      <Group gap="xs" mb="sm">
        <IconChartHistogram size={18} />
        <Text fw={600}>Spectrogram</Text>
        <Text size="sm" c="dimmed">
          Time-frequency view
        </Text>
      </Group>
      <div className="plot-container">
        <SpectrogramCanvas row={row} />
      </div>
    </Paper>
  );
}

export default SpectrogramPanel;
