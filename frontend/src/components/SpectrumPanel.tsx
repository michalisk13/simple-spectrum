import { Badge, Group, Paper, Text } from "@mantine/core";
import { IconWaveSine } from "@tabler/icons-react";
import { useCallback, useMemo, useRef, useState, type MutableRefObject } from "react";
import type { EngineStatus } from "../api/types";
import { useAnimationFrame } from "../hooks/useAnimationFrame";
import type { SpectrumFrame, SpectrumMetaFrame } from "../ws/types";
import SpectrumCanvas from "./plots/SpectrumCanvas";

const formatHz = (value: number) => {
  if (!Number.isFinite(value)) {
    return "--";
  }
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(3)} GHz`;
  }
  if (value >= 1e6) {
    return `${(value / 1e6).toFixed(2)} MHz`;
  }
  if (value >= 1e3) {
    return `${(value / 1e3).toFixed(1)} kHz`;
  }
  return `${value.toFixed(0)} Hz`;
};

type SpectrumPanelProps = {
  statusFrame: EngineStatus | null;
  spectrumFrameRef: MutableRefObject<SpectrumFrame | null>;
};

const shouldUpdateSpectrumMeta = (
  previous: SpectrumMetaFrame | null,
  next: SpectrumMetaFrame,
) => {
  if (!previous) {
    return true;
  }
  return (
    previous.freq_start_hz !== next.freq_start_hz ||
    previous.freq_stop_hz !== next.freq_stop_hz ||
    previous.rbw_hz !== next.rbw_hz ||
    previous.vbw_hz !== next.vbw_hz ||
    previous.fft_size !== next.fft_size ||
    previous.detector !== next.detector
  );
};

function SpectrumPanel({ statusFrame, spectrumFrameRef }: SpectrumPanelProps) {
  const [meta, setMeta] = useState<SpectrumMetaFrame | null>(null);
  const [hasTrace, setHasTrace] = useState(false);
  const lastSeqRef = useRef<number | null>(null);
  const metaRef = useRef<SpectrumMetaFrame | null>(null);

  const updateTrace = useCallback(() => {
    const frame = spectrumFrameRef.current;
    if (!frame) {
      if (lastSeqRef.current !== null) {
        lastSeqRef.current = null;
        metaRef.current = null;
        setMeta(null);
        setHasTrace(false);
      }
      return;
    }

    if (frame.meta.seq !== lastSeqRef.current) {
      lastSeqRef.current = frame.meta.seq;
      setHasTrace(frame.payload.length > 0);
      if (shouldUpdateSpectrumMeta(metaRef.current, frame.meta)) {
        metaRef.current = frame.meta;
        setMeta(frame.meta);
      }
    }
  }, [spectrumFrameRef]);

  useAnimationFrame(updateTrace, 20);

  const spectrumInfo = useMemo(() => {
    if (!meta) {
      return null;
    }
    const center = (meta.freq_start_hz + meta.freq_stop_hz) / 2;
    const span = meta.freq_stop_hz - meta.freq_start_hz;
    return {
      center,
      span,
      rbw: meta.rbw_hz,
      vbw: meta.vbw_hz,
      fftSize: meta.fft_size,
      detector: meta.detector,
    };
  }, [meta]);

  const isConnected = statusFrame?.connected ?? false;
  const showDisconnected = !isConnected && !hasTrace;
  const showAwaiting = isConnected && !hasTrace;

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
          <Badge variant="light">
            FFT {spectrumInfo ? spectrumInfo.fftSize : "--"}
          </Badge>
          <Badge variant="light">
            {spectrumInfo ? `${formatHz(spectrumInfo.rbw)} RBW` : "RBW --"}
          </Badge>
        </Group>
      </Group>
      <Group gap="lg" className="panel-meta">
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Center
          </Text>
          <Text size="sm" fw={500}>
            {spectrumInfo ? formatHz(spectrumInfo.center) : "--"}
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Span
          </Text>
          <Text size="sm" fw={500}>
            {spectrumInfo ? formatHz(spectrumInfo.span) : "--"}
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            RBW / VBW
          </Text>
          <Text size="sm" fw={500}>
            {spectrumInfo
              ? `${formatHz(spectrumInfo.rbw)} / ${formatHz(spectrumInfo.vbw)}`
              : "--"}
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Ref / Range
          </Text>
          <Text size="sm" fw={500}>
            --
          </Text>
        </div>
        <div className="panel-meta-item">
          <Text size="xs" c="dimmed">
            Detector
          </Text>
          <Text size="sm" fw={500}>
            {spectrumInfo ? spectrumInfo.detector : "--"}
          </Text>
        </div>
      </Group>
      <div className="plot-container">
        <SpectrumCanvas frameRef={spectrumFrameRef} />
        {showDisconnected ? (
          <div className="plot-overlay">No data / disconnected</div>
        ) : null}
        {showAwaiting ? (
          <div className="plot-overlay">Awaiting spectrum stream…</div>
        ) : null}
      </div>
    </Paper>
  );
}

export default SpectrumPanel;
