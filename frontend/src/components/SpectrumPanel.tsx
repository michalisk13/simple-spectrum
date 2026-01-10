import { Group, Paper, Text } from "@mantine/core";
import { IconWaveSine } from "@tabler/icons-react";
import { useCallback, useRef, useState } from "react";
import SpectrumCanvas from "./plots/SpectrumCanvas";
import { useAnimationFrame } from "../hooks/useAnimationFrame";
import { generateSpectrumTrace } from "../utils/mockData";

const TRACE_POINTS = 512;

function SpectrumPanel() {
  // Track mock phase so the trace animates smoothly over time.
  const phaseRef = useRef(0);
  // Store the current trace in state to trigger a canvas redraw.
  const [trace, setTrace] = useState(() =>
    generateSpectrumTrace(TRACE_POINTS, phaseRef.current),
  );

  const updateTrace = useCallback((deltaMs: number) => {
    phaseRef.current += deltaMs * 0.002;
    setTrace(generateSpectrumTrace(TRACE_POINTS, phaseRef.current));
  }, []);

  // Drive the mock spectrum at 20 fps to validate rendering performance.
  useAnimationFrame(updateTrace, 20);

  return (
    <Paper className="panel-surface" radius="lg" p="md">
      <Group gap="xs" mb="sm">
        <IconWaveSine size={18} />
        <Text fw={600}>Spectrum</Text>
        <Text size="sm" c="dimmed">
          Live FFT trace
        </Text>
      </Group>
      <div className="plot-container">
        <SpectrumCanvas trace={trace} />
      </div>
    </Paper>
  );
}

export default SpectrumPanel;
