import {
  Badge,
  Box,
  Button,
  Checkbox,
  Divider,
  Group,
  NumberInput,
  Paper,
  SegmentedControl,
  Select,
  Stack,
  Switch,
  Text,
  TextInput,
} from "@mantine/core";
import {
  IconAdjustments,
  IconAdjustmentsHorizontal,
  IconChartHistogram,
  IconRadio,
  IconStar,
  IconTarget,
  IconWaveSine,
} from "@tabler/icons-react";

const sectionHeaderProps = {
  size: "xs",
  fw: 600,
  tt: "uppercase",
  c: "dimmed",
  className: "sidebar-section-title",
} as const;

function LeftSidebar() {
  return (
    <Stack gap="lg" className="sidebar">
      <Group justify="space-between" align="center">
        <Group gap="sm">
          <IconRadio size={20} />
          <Text fw={600} size="lg">
            Controls
          </Text>
        </Group>
        <Badge variant="light" color="red">
          Offline
        </Badge>
      </Group>

      <Box>
        <Group gap="xs" mb="sm">
          <IconWaveSine size={16} />
          <Text {...sectionHeaderProps}>Controls</Text>
        </Group>
        <Paper withBorder className="sidebar-section">
          <Stack gap="sm">
            <Group gap="xs" grow>
              <Button size="xs" variant="light">
                Run
              </Button>
              <Button size="xs" variant="outline">
                Single
              </Button>
            </Group>
            <Group gap="xs" grow>
              <Button size="xs" variant="subtle">
                Save trace CSV
              </Button>
              <Button size="xs" variant="subtle">
                Calibration
              </Button>
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Detector
              </Text>
              <Select
                data={["RMS", "Peak", "Sample"]}
                defaultValue="RMS"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Trace mode
              </Text>
              <Select
                data={["Clear Write", "Max Hold", "Min Hold", "Average"]}
                defaultValue="Clear Write"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Avg count
              </Text>
              <NumberInput
                defaultValue={10}
                min={1}
                max={50}
                size="xs"
                variant="filled"
                hideControls
              />
            </Group>
            <Group className="sidebar-switches" gap="xs">
              <Checkbox label="Auto ref level" defaultChecked size="xs" />
              <Checkbox label="Hold across settings" size="xs" />
              <Checkbox label="DC remove" defaultChecked size="xs" />
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
              <Button size="xs" variant="light" fullWidth>
                Fast View
              </Button>
              <Button size="xs" variant="outline" fullWidth>
                Wide Scan
              </Button>
              <Button size="xs" variant="outline" fullWidth>
                Measure
              </Button>
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Measure detector
              </Text>
              <Select
                data={["Peak", "RMS"]}
                defaultValue="Peak"
                size="xs"
                variant="filled"
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
              <TextInput
                label="Center (Hz)"
                defaultValue="2.437e9"
                size="xs"
                variant="filled"
              />
              <TextInput
                label="Span (Hz)"
                defaultValue="20e6"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                RBW
              </Text>
              <Select
                data={["Manual 4.9 kHz", "Auto", "1 kHz", "10 kHz"]}
                defaultValue="Manual 4.9 kHz"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                VBW
              </Text>
              <Select
                data={["Auto 3 kHz", "1 kHz", "3 kHz", "10 kHz"]}
                defaultValue="Auto 3 kHz"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Window
              </Text>
              <Select
                data={["Blackman Harris", "Hann", "Flat Top"]}
                defaultValue="Blackman Harris"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Marker controls
              </Text>
              <SegmentedControl
                size="xs"
                fullWidth
                data={[
                  { label: "Peak", value: "peak" },
                  { label: "Next ◀", value: "prev" },
                  { label: "Next ▶", value: "next" },
                ]}
              />
            </Group>
            <Group gap="xs" grow>
              <Button size="xs" variant="subtle" leftSection={<IconTarget size={12} />}>
                Marker to center
              </Button>
              <Button size="xs" variant="subtle">
                Clear
              </Button>
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
                data={["Manual", "Fast Attack", "Slow Attack"]}
                defaultValue="Manual"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Gain (dB)
              </Text>
              <NumberInput
                defaultValue={55}
                min={0}
                max={70}
                size="xs"
                variant="filled"
                hideControls
              />
            </Group>
            <Group className="sidebar-grid">
              <NumberInput
                label="Ref level (dB)"
                defaultValue={0}
                size="xs"
                variant="filled"
                hideControls
              />
              <NumberInput
                label="Range (dB)"
                defaultValue={100}
                size="xs"
                variant="filled"
                hideControls
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Scale
              </Text>
              <Select
                data={["Linear", "Log", "dBFS"]}
                defaultValue="dBFS"
                size="xs"
                variant="filled"
              />
            </Group>
            <Switch label="Show trace 2" size="xs" />
            <Checkbox label="Hold max trace" size="xs" />
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
                defaultValue="PSD (Welch)"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-grid">
              <NumberInput
                label="Time res (s/s)"
                defaultValue={15}
                size="xs"
                variant="filled"
                hideControls
              />
              <NumberInput
                label="Time span (s)"
                defaultValue={20}
                size="xs"
                variant="filled"
                hideControls
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Colormap
              </Text>
              <Select
                data={["viridis", "plasma"]}
                defaultValue="viridis"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-row" align="center">
              <Text size="xs" c="dimmed">
                Scale
              </Text>
              <Select
                data={["Auto (5-95%)", "Fixed floor", "Manual"]}
                defaultValue="Auto (5-95%)"
                size="xs"
                variant="filled"
              />
            </Group>
            <Group className="sidebar-grid">
              <NumberInput
                label="dB min"
                defaultValue={-120}
                size="xs"
                variant="filled"
                hideControls
              />
              <NumberInput
                label="dB max"
                defaultValue={0}
                size="xs"
                variant="filled"
                hideControls
              />
            </Group>
            <Switch label="Auto range (±2σ)" size="xs" defaultChecked />
          </Stack>
        </Paper>
      </Box>
    </Stack>
  );
}

export default LeftSidebar;
