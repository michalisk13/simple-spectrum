// Mock data generators for spectrum and spectrogram scaffolds.

// Generate a smooth spectrum trace with a couple of sinusoidal peaks.
export const generateSpectrumTrace = (points: number, phase: number) => {
  const trace: number[] = [];
  const peak1 = Math.sin(phase * 0.6) * 0.35 + 0.6;
  const peak2 = Math.cos(phase * 0.35) * 0.25 + 0.4;

  for (let i = 0; i < points; i += 1) {
    const x = i / (points - 1);
    const ridge = Math.sin(x * Math.PI * 3 + phase) * 0.15;
    const peakA = Math.exp(-Math.pow((x - 0.28) * 8, 2)) * peak1;
    const peakB = Math.exp(-Math.pow((x - 0.7) * 10, 2)) * peak2;
    const noise = (Math.sin(x * 120 + phase * 1.2) + 1) * 0.03;

    trace.push(Math.max(0, ridge + peakA + peakB + noise + 0.05));
  }

  return trace;
};

// Generate a spectrogram row with drifting energy bands.
export const generateSpectrogramRow = (points: number, phase: number) => {
  const row: number[] = [];
  const bandCenter = (Math.sin(phase * 0.25) + 1) * 0.35 + 0.1;
  const bandWidth = 0.08 + (Math.cos(phase * 0.2) + 1) * 0.05;

  for (let i = 0; i < points; i += 1) {
    const x = i / (points - 1);
    const band = Math.exp(-Math.pow((x - bandCenter) / bandWidth, 2));
    const harmonic = Math.sin(x * 12 + phase * 1.4) * 0.2 + 0.2;
    const noise = (Math.cos(x * 90 + phase * 0.9) + 1) * 0.08;

    row.push(Math.min(1, band + harmonic + noise));
  }

  return row;
};
