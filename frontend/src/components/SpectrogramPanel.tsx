import { Badge, Group, Paper, Text } from "@mantine/core";
import { IconChartHistogram } from "@tabler/icons-react";
import { useCallback, useMemo, useRef, useState, type MutableRefObject } from "react";
import type { EngineStatus } from "../api/types";
import { useAnimationFrame } from "../hooks/useAnimationFrame";
import type { SpectrogramFrame, SpectrogramMetaFrame } from "../ws/types";
import SpectrogramCanvas from "./plots/SpectrogramCanvas";

const formatDb = (value: number) => {
  if (!Number.isFinite(value)) {
    return "--";
  }
  return `${value.toFixed(0)} dB`;
};

type SpectrogramPanelProps = {
  statusFrame: EngineStatus | null;
  spectrogramFrameRef: MutableRefObject<SpectrogramFrame | null>;
};

const shouldUpdateSpectrogramMeta = (
  previous: SpectrogramMetaFrame | null,
  next: SpectrogramMetaFrame,
) => {
  if (!previous) {
    return true;
  }
  return (
    previous.colormap !== next.colormap ||
    previous.db_min !== next.db_min ||
    previous.db_max !== next.db_max ||
    previous.quantized !== next.quantized ||
    previous.dtype !== next.dtype
  );
};

function SpectrogramPanel({
  statusFrame,
  spectrogramFrameRef,
}: SpectrogramPanelProps) {
  const [meta, setMeta] = useState<SpectrogramMetaFrame | null>(null);
  const [hasRow, setHasRow] = useState(false);
  const lastSeqRef = useRef<number | null>(null);
  const metaRef = useRef<SpectrogramMetaFrame | null>(null);

  const updateRow = useCallback(() => {
    const frame = spectrogramFrameRef.current;
    if (!frame) {
      if (lastSeqRef.current !== null) {
        lastSeqRef.current = null;
        metaRef.current = null;
        setMeta(null);
        setHasRow(false);
      }
      return;
    }

    if (frame.meta.seq !== lastSeqRef.current) {
      lastSeqRef.current = frame.meta.seq;
      setHasRow(frame.payload.length > 0);
      if (shouldUpdateSpectrogramMeta(metaRef.current, frame.meta)) {
        metaRef.current = frame.meta;
        setMeta(frame.meta);
      }
    }
  }, [spectrogramFrameRef]);

  useAnimationFrame(updateRow, 20);

  const spectrogramInfo = useMemo(() => {
    if (!meta) {
      return null;
    }
    return {
      colormap: meta.colormap,
      dbMin: meta.db_min,
      dbMax: meta.db_max,
      quantized: meta.quantized,
      dtype: meta.dtype,
    };
  }, [meta]);

  const isConnected = statusFrame?.connected ?? false;
  const showDisconnected = !isConnected && !hasRow;
  const showAwaiting = isConnected && !hasRow;

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
          <Badge variant="light">
            {spectrogramInfo
              ? `Mode: ${spectrogramInfo.quantized ? "Quantized" : "Float"}`
              : "Mode: --"}
          </Badge>
          <Badge variant="light">
            {statusFrame
              ? `${statusFrame.spectrogram_rate.toFixed(1)} slices/s`
              : "Slices/s --"}
          </Badge>
        </Group>
      </Group>
      <Group gap="lg" className="panel-meta">
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Time span
          </Text>
          <Text size="sm" fw={500}>
            {statusFrame
              ? `${statusFrame.spectrogram_time_span_s.toFixed(1)} s`
              : "--"}
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Colormap
          </Text>
          <Text size="sm" fw={500}>
            {spectrogramInfo ? spectrogramInfo.colormap : "--"}
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Range
          </Text>
          <Text size="sm" fw={500}>
            {spectrogramInfo
              ? `${formatDb(spectrogramInfo.dbMin)} to ${formatDb(
                  spectrogramInfo.dbMax,
                )}`
              : "--"}
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Auto range
          </Text>
          <Text size="sm" fw={500}>
            --
          </Text>
        </div>
      </Group>
      <div className="plot-container">
        <SpectrogramCanvas frameRef={spectrogramFrameRef} />
        {showDisconnected ? (
          <div className="plot-overlay">No data / disconnected</div>
        ) : null}
        {showAwaiting ? (
          <div className="plot-overlay">Awaiting spectrogram stream…</div>
        ) : null}
      </div>
    </Paper>
  );
}

export default SpectrogramPanel;
