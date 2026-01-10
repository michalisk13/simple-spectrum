import { Badge, Group, Paper, Text } from "@mantine/core";
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
      <Group justify="space-between" align="center" className="panel-header">
        <Group gap="xs">
          <IconWaveSine size={18} />
          <Text fw={600}>Spectrum</Text>
          <Text size="sm" c="dimmed">
            Live FFT trace
          </Text>
        </Group>
        <Group gap="xs">
          <Badge variant="light">FFT 8192</Badge>
          <Badge variant="light">Update 100 ms</Badge>
        </Group>
      </Group>
      <Group gap="lg" className="panel-meta">
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Center
          </Text>
          <Text size="sm" fw={500}>
            2.437 GHz
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Span
          </Text>
          <Text size="sm" fw={500}>
            20 MHz
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            RBW / VBW
          </Text>
          <Text size="sm" fw={500}>
            4.9 kHz / 3 kHz
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Ref / Range
          </Text>
          <Text size="sm" fw={500}>
            0 dB / 100 dB
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Detector
          </Text>
          <Text size="sm" fw={500}>
            RMS
          </Text>
        </div>
      </Group>
      <div className="plot-container">
        <SpectrumCanvas trace={trace} />
      </div>
    </Paper>
  );
}

export default SpectrumPanel;
