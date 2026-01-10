import { useEffect, useMemo, useRef } from "react";

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export type SpectrumCanvasProps = {
  trace: number[];
};

const SpectrumCanvas = ({ trace }: SpectrumCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const traceStats = useMemo(() => {
    const max = trace.length ? Math.max(...trace) : 1;
    return { max: max || 1 };
  }, [trace]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const updateCanvasSize = () => {
      const parent = canvas.parentElement;
      if (!parent) {
        return;
      }

      const { width, height } = parent.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };

    updateCanvasSize();

    resizeObserverRef.current = new ResizeObserver(() => {
      updateCanvasSize();
    });

    resizeObserverRef.current.observe(canvas.parentElement as Element);

    return () => {
      resizeObserverRef.current?.disconnect();
      resizeObserverRef.current = null;
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const width = canvas.width;
    const height = canvas.height;

    // Paint the background and grid.
    ctx.fillStyle = "#0b1220";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
    ctx.lineWidth = 1;
    const gridLines = 4;
    for (let i = 1; i <= gridLines; i += 1) {
      const y = (height / (gridLines + 1)) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!trace.length) {
      return;
    }

    const maxValue = traceStats.max;

    // Draw the spectrum polyline scaled to canvas dimensions.
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;
    ctx.beginPath();

    const denominator = trace.length > 1 ? trace.length - 1 : 1;

    trace.forEach((value, index) => {
      const x = (index / denominator) * (width - 1);
      const normalized = clamp(value / maxValue, 0, 1);
      const y = height - 1 - normalized * (height - 4) - 2;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Add a subtle glow to mimic a live trace.
    ctx.shadowColor = "rgba(56, 189, 248, 0.4)";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [trace, traceStats]);

  return <canvas ref={canvasRef} className="plot-canvas" />;
};

export default SpectrumCanvas;
