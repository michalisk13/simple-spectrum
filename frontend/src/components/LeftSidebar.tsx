import {
  Badge,
  Box,
  Button,
  Divider,
  Group,
  NumberInput,
  Paper,
  Select,
  Stack,
  Switch,
  Text,
} from "@mantine/core";
import {
  IconAdjustments,
  IconAdjustmentsHorizontal,
  IconChartHistogram,
  IconRadio,
  IconStar,
  IconWaveSine,
} from "@tabler/icons-react";
import { useEffect, useMemo, useState } from "react";
import { ApiClient } from "../api/client";
import type {
  ApiConfigResponse,
  ApplyPresetRequest,
  ApplyPresetResponse,
  ConfigUpdatePayload,
  PresetName,
  SpectrumConfig,
  StreamMetadata,
} from "../api/types";

type LeftSidebarProps = {
  connectionState: "connected" | "disconnected" | "checking";
  config: SpectrumConfig | null;
  stream: StreamMetadata | null;
  isConfigLoading: boolean;
  onUpdateConfig: (payload: ConfigUpdatePayload) => Promise<ApiConfigResponse | null>;
  onApplyPreset: (payload: ApplyPresetRequest) => Promise<ApplyPresetResponse | null>;
};

const sectionHeaderProps = {
  size: "xs",
  fw: 600,
  tt: "uppercase",
  c: "dimmed",
  className: "sidebar-section-title",
} as const;

const fallbackPresets: PresetName[] = ["Fast View", "Wide Scan", "Measure"];

type FormState = {
  centerHz: number;
  spanHz: number;
  rbwMode: string;
  rbwHz: number;
  vbwMode: string;
  vbwHz: number;
  window: string;
  detector: string;
  traceType: string;
  avgCount: number;
  avgMode: string;
  measurementMode: boolean;
  dcRemove: boolean;
  gainMode: string;
  gainDb: number;
  refLevelDb: number;
  displayRangeDb: number;
  trace2Enabled: boolean;
  spectrogramEnabled: boolean;
  spectrogramRate: number;
  spectrogramTimeSpan: number;
  spectrogramMode: string;
  spectrogramCmap: string;
  spectrogramMinDb: number;
  spectrogramMaxDb: number;
};

const buildFormState = (
  config: SpectrumConfig,
  stream: StreamMetadata,
): FormState => ({
  centerHz: config.center_hz,
  spanHz: config.sample_rate_hz,
  rbwMode: config.rbw_mode,
  rbwHz: config.rbw_hz,
  vbwMode: config.vbw_mode,
  vbwHz: config.vbw_hz,
  window: config.window,
  detector: config.detector,
  traceType: config.trace_type,
  avgCount: config.avg_count,
  avgMode: config.avg_mode,
  measurementMode: config.measurement_mode,
  dcRemove: config.dc_remove,
  gainMode: config.gain_mode,
  gainDb: config.gain_db,
  refLevelDb: config.ref_level_db,
  displayRangeDb: config.display_range_db,
  trace2Enabled: config.trace2_enabled,
  spectrogramEnabled: stream.spectrogram_enabled,
  spectrogramRate: stream.spectrogram_rate,
  spectrogramTimeSpan: stream.spectrogram_time_span_s,
  spectrogramMode: config.spectrogram_mode,
  spectrogramCmap: stream.spectrogram_cmap,
  spectrogramMinDb: stream.spectrogram_min_db,
  spectrogramMaxDb: stream.spectrogram_max_db,
});

function LeftSidebar({
  connectionState,
  config,
  stream,
  isConfigLoading,
  onUpdateConfig,
  onApplyPreset,
}: LeftSidebarProps) {
  const apiClient = useMemo(() => new ApiClient(), []);
  const [presets, setPresets] = useState<PresetName[]>(fallbackPresets);
  const [isLoadingPresets, setIsLoadingPresets] = useState(false);
  const [activePreset, setActivePreset] = useState<PresetName | null>(null);
  const [measureDetector, setMeasureDetector] = useState("Peak");
  const [formState, setFormState] = useState<FormState | null>(null);
  const [isSavingConfig, setIsSavingConfig] = useState(false);

  useEffect(() => {
    const loadPresets = async () => {
      setIsLoadingPresets(true);
      const response = await apiClient.listPresets();
      if (response?.presets?.length) {
        setPresets(response.presets);
      }
      setIsLoadingPresets(false);
    };

    void loadPresets();
  }, [apiClient]);

  useEffect(() => {
    if (!config || !stream) {
      setFormState(null);
      return;
    }
    setFormState(buildFormState(config, stream));
  }, [config, stream]);

  const applyPreset = async (payload: ApplyPresetRequest) => {
    if (activePreset) {
      return;
    }
    setActivePreset(payload.name);
    try {
      const response = await onApplyPreset(payload);
      if (!response && config && stream) {
        setFormState(buildFormState(config, stream));
      }
    } finally {
      setActivePreset(null);
    }
  };

  const renderPresetButton = (presetName: PresetName, variant: "light" | "outline") => (
    <Button
      key={presetName}
      size="xs"
      variant={variant}
      fullWidth
      loading={isLoadingPresets || activePreset === presetName}
      disabled={isLoadingPresets || Boolean(activePreset)}
      onClick={() =>
        applyPreset({
          name: presetName,
          measure_detector: presetName === "Measure" ? measureDetector : undefined,
        })
      }
    >
      {presetName}
    </Button>
  );

  const updateFormField = <Key extends keyof FormState>(
    key: Key,
    value: FormState[Key],
  ) => {
    setFormState((prev) => (prev ? { ...prev, [key]: value } : prev));
  };

  const submitConfigUpdate = async (payload: ConfigUpdatePayload) => {
    if (isSavingConfig) {
      return;
    }
    setIsSavingConfig(true);
    const response = await onUpdateConfig(payload);
    setIsSavingConfig(false);
    if (!response && config && stream) {
      setFormState(buildFormState(config, stream));
    }
  };

  const isConfigReady = Boolean(formState);
  const controlsDisabled = isConfigLoading || isSavingConfig || !isConfigReady;
  const spectrogramControlsDisabled =
    controlsDisabled || !formState?.spectrogramEnabled;

  return (
    <Stack gap="lg" className="sidebar">
      <Group justify="space-between" align="center">
        <Group gap="sm">
          <IconRadio size={20} />
          <Text fw={600} size="lg">
            Controls
          </Text>
        </Group>
        <Badge
          variant="light"
          color={
            connectionState === "connected"
              ? "green"
              : connectionState === "checking"
                ? "yellow"
                : "red"
          }
        >
          {connectionState === "connected"
            ? "Online"
            : connectionState === "checking"
              ? "Checking"
              : "Offline"}
        </Badge>
      </Group>

      <Box>
        <Group gap="xs" mb="sm">
          <IconWaveSine size={16} />
          <Text {...sectionHeaderProps}>Trace</Text>
        </Group>
        <Paper withBorder className="sidebar-section">
          <Stack gap="sm">
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Detector
              </Text>
              <Select
                data={["RMS", "Peak", "Sample"]}
                value={formState?.detector ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("detector", value);
                  void submitConfigUpdate({ detector: value });
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Trace mode
              </Text>
              <Select
                data={["Clear Write", "Max Hold", "Min Hold", "Average"]}
                value={formState?.traceType ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("traceType", value);
                  void submitConfigUpdate({ trace_type: value });
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Avg count
              </Text>
              <NumberInput
                value={formState?.avgCount ?? 0}
                min={1}
                max={50}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("avgCount", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({ avg_count: formState.avgCount });
                  }
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Avg mode
              </Text>
              <Select
                data={["RMS", "Log"]}
                value={formState?.avgMode ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("avgMode", value);
                  void submitConfigUpdate({ avg_mode: value });
                }}
              />
            </Group>
            <Group className="sidebar-switches" gap="xs">
              <Switch
                label="Measurement mode"
                size="xs"
                checked={formState?.measurementMode ?? false}
                disabled={controlsDisabled}
                onChange={(event) => {
                  const nextValue = event.currentTarget.checked;
                  updateFormField("measurementMode", nextValue);
                  void submitConfigUpdate({ measurement_mode: nextValue });
                }}
              />
              <Switch
                label="DC remove"
                size="xs"
                checked={formState?.dcRemove ?? false}
                disabled={controlsDisabled}
                onChange={(event) => {
                  const nextValue = event.currentTarget.checked;
                  updateFormField("dcRemove", nextValue);
                  void submitConfigUpdate({ dc_remove: nextValue });
                }}
              />
            </Group>
          </Stack>
        </Paper>
      </Box>

      <Divider />

      <Box>
        <Group gap="xs" mb="sm">
          <IconStar size={16} />
          <Text {...sectionHeaderProps}>Presets</Text>
        </Group>
        <Paper withBorder className="sidebar-section">
          <Stack gap="sm">
            <Group gap="xs">
              {presets.length >= 3 ? (
                presets.slice(0, 3).map((preset, index) =>
                  renderPresetButton(preset, index === 0 ? "light" : "outline"),
                )
              ) : (
                <>
                  {renderPresetButton("Fast View", "light")}
                  {renderPresetButton("Wide Scan", "outline")}
                  {renderPresetButton("Measure", "outline")}
                </>
              )}
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Measure detector
              </Text>
              <Select
                data={["Peak", "RMS"]}
                value={measureDetector}
                onChange={(value) => {
                  if (value) {
                    setMeasureDetector(value);
                  }
                }}
                size="xs"
                variant="filled"
                disabled={isLoadingPresets || Boolean(activePreset)}
              />
            </Group>
          </Stack>
        </Paper>
      </Box>

      <Divider />

      <Box>
        <Group gap="xs" mb="sm">
          <IconAdjustmentsHorizontal size={16} />
          <Text {...sectionHeaderProps}>Frequency</Text>
        </Group>
        <Paper withBorder className="sidebar-section">
          <Stack gap="sm">
            <Group className="sidebar-grid">
              <NumberInput
                label="Center (Hz)"
                value={formState?.centerHz ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("centerHz", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({ center_hz: formState.centerHz });
                  }
                }}
              />
              <NumberInput
                label="Span (Hz)"
                value={formState?.spanHz ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("spanHz", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({ span_hz: formState.spanHz });
                  }
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                RBW
              </Text>
              <Select
                data={["Manual", "Auto"]}
                value={formState?.rbwMode ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("rbwMode", value);
                  void submitConfigUpdate({ rbw_mode: value });
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <NumberInput
                label="RBW (Hz)"
                value={formState?.rbwHz ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("rbwHz", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({ rbw_hz: formState.rbwHz });
                  }
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                VBW
              </Text>
              <Select
                data={["Manual", "Auto"]}
                value={formState?.vbwMode ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("vbwMode", value);
                  void submitConfigUpdate({ vbw_mode: value });
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <NumberInput
                label="VBW (Hz)"
                value={formState?.vbwHz ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("vbwHz", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({ vbw_hz: formState.vbwHz });
                  }
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Window
              </Text>
              <Select
                data={["Blackman Harris", "Hann", "Flat Top"]}
                value={formState?.window ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("window", value);
                  void submitConfigUpdate({ window: value });
                }}
              />
            </Group>
          </Stack>
        </Paper>
      </Box>

      <Divider />

      <Box>
        <Group gap="xs" mb="sm">
          <IconAdjustments size={16} />
          <Text {...sectionHeaderProps}>Gain &amp; Display</Text>
        </Group>
        <Paper withBorder className="sidebar-section">
          <Stack gap="sm">
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Gain mode
              </Text>
              <Select
                data={[
                  { label: "Manual", value: "manual" },
                  { label: "Fast Attack", value: "fast_attack" },
                  { label: "Slow Attack", value: "slow_attack" },
                  { label: "Hybrid", value: "hybrid" },
                ]}
                value={formState?.gainMode ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("gainMode", value);
                  void submitConfigUpdate({ gain_mode: value });
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Gain (dB)
              </Text>
              <NumberInput
                value={formState?.gainDb ?? 0}
                min={0}
                max={70}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("gainDb", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({ gain_db: formState.gainDb });
                  }
                }}
              />
            </Group>
            <Group className="sidebar-grid">
              <NumberInput
                label="Ref level (dB)"
                value={formState?.refLevelDb ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("refLevelDb", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({ ref_level_db: formState.refLevelDb });
                  }
                }}
              />
              <NumberInput
                label="Range (dB)"
                value={formState?.displayRangeDb ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("displayRangeDb", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({
                      display_range_db: formState.displayRangeDb,
                    });
                  }
                }}
              />
            </Group>
            <Switch
              label="Show trace 2"
              size="xs"
              checked={formState?.trace2Enabled ?? false}
              disabled={controlsDisabled}
              onChange={(event) => {
                const nextValue = event.currentTarget.checked;
                updateFormField("trace2Enabled", nextValue);
                void submitConfigUpdate({ trace2_enabled: nextValue });
              }}
            />
          </Stack>
        </Paper>
      </Box>

      <Divider />

      <Box>
        <Group gap="xs" mb="sm">
          <IconChartHistogram size={16} />
          <Text {...sectionHeaderProps}>Spectrogram</Text>
        </Group>
        <Paper withBorder className="sidebar-section">
          <Stack gap="sm">
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Mode
              </Text>
              <Select
                data={["PSD (Welch)", "Peak Hold"]}
                value={formState?.spectrogramMode ?? null}
                size="xs"
                variant="filled"
                disabled={controlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("spectrogramMode", value);
                  void submitConfigUpdate({ spectrogram_mode: value });
                }}
              />
            </Group>
            <Group className="sidebar-grid">
              <NumberInput
                label="Time res (s/s)"
                value={formState?.spectrogramRate ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={spectrogramControlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("spectrogramRate", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({
                      spectrogram_rate: formState.spectrogramRate,
                    });
                  }
                }}
              />
              <NumberInput
                label="Time span (s)"
                value={formState?.spectrogramTimeSpan ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={spectrogramControlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("spectrogramTimeSpan", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({
                      spectrogram_time_span_s: formState.spectrogramTimeSpan,
                    });
                  }
                }}
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Colormap
              </Text>
              <Select
                data={["viridis", "plasma"]}
                value={formState?.spectrogramCmap ?? null}
                size="xs"
                variant="filled"
                disabled={spectrogramControlsDisabled}
                onChange={(value) => {
                  if (!value) {
                    return;
                  }
                  updateFormField("spectrogramCmap", value);
                  void submitConfigUpdate({ spectrogram_cmap: value });
                }}
              />
            </Group>
            <Group className="sidebar-grid">
              <NumberInput
                label="dB min"
                value={formState?.spectrogramMinDb ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={spectrogramControlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("spectrogramMinDb", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({
                      spectrogram_min_db: formState.spectrogramMinDb,
                    });
                  }
                }}
              />
              <NumberInput
                label="dB max"
                value={formState?.spectrogramMaxDb ?? 0}
                size="xs"
                variant="filled"
                hideControls
                disabled={spectrogramControlsDisabled}
                onChange={(value) => {
                  if (typeof value === "number") {
                    updateFormField("spectrogramMaxDb", value);
                  }
                }}
                onBlur={() => {
                  if (formState) {
                    void submitConfigUpdate({
                      spectrogram_max_db: formState.spectrogramMaxDb,
                    });
                  }
                }}
              />
            </Group>
            <Switch
              label="Enable spectrogram"
              size="xs"
              checked={formState?.spectrogramEnabled ?? false}
              disabled={controlsDisabled}
              onChange={(event) => {
                const nextValue = event.currentTarget.checked;
                updateFormField("spectrogramEnabled", nextValue);
                void submitConfigUpdate({ spectrogram_enabled: nextValue });
              }}
            />
          </Stack>
        </Paper>
      </Box>
    </Stack>
  );
}

export default LeftSidebar;
