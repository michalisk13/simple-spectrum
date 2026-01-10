import { Badge, Group, Paper, Text } from "@mantine/core";
import { IconChartHistogram } from "@tabler/icons-react";
import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import type { EngineStatus } from "../api/types";
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
  spectrogramFrame: SpectrogramFrame | null;
  spectrogramFrameRef: MutableRefObject<SpectrogramFrame | null>;
};

function SpectrogramPanel({
  statusFrame,
  spectrogramFrame,
  spectrogramFrameRef,
}: SpectrogramPanelProps) {
  const [row, setRow] = useState<Float32Array | Uint8Array | null>(null);
  const [meta, setMeta] = useState<SpectrogramMetaFrame | null>(null);
  const lastSeqRef = useRef<number | null>(null);

  useEffect(() => {
    const frame = spectrogramFrame ?? spectrogramFrameRef.current;
    if (!frame) {
      if (lastSeqRef.current !== null) {
        lastSeqRef.current = null;
        setRow(null);
        setMeta(null);
      }
      return;
    }

    if (frame.meta.seq !== lastSeqRef.current) {
      lastSeqRef.current = frame.meta.seq;
      setRow(frame.payload);
      setMeta(frame.meta);
    }
  }, [spectrogramFrame, spectrogramFrameRef]);

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
  const hasRow = Boolean(row && row.length);

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
        {!isConnected || !hasRow ? (
          <div className="panel-placeholder">Awaiting spectrogram stream…</div>
        ) : (
          <SpectrogramCanvas row={row ?? new Float32Array()} meta={meta} />
        )}
      </div>
    </Paper>
  );
}

export default SpectrogramPanel;
