import { useEffect, useMemo, useRef } from "react";

const palette = (value: number) => {
  const clamped = Math.min(1, Math.max(0, value));
  const r = Math.floor(20 + clamped * 80);
  const g = Math.floor(40 + clamped * 140);
  const b = Math.floor(120 + clamped * 120);

  return { r, g, b };
};

export type SpectrogramCanvasProps = {
  row: number[];
};

const SpectrogramCanvas = ({ row }: SpectrogramCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bufferRef = useRef<HTMLCanvasElement | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);

  const rowLength = useMemo(() => row.length, [row.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    if (!bufferRef.current) {
      bufferRef.current = document.createElement("canvas");
    }

    const buffer = bufferRef.current;

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

      buffer.width = Math.max(1, Math.floor(width));
      buffer.height = Math.max(1, Math.floor(height));
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
    const buffer = bufferRef.current;
    if (!canvas || !buffer || rowLength === 0) {
      return;
    }

    const bufferCtx = buffer.getContext("2d");
    const ctx = canvas.getContext("2d");
    if (!bufferCtx || !ctx) {
      return;
    }

    const bufferWidth = buffer.width;
    const bufferHeight = buffer.height;
    const widthScale = bufferWidth > 1 ? bufferWidth - 1 : 1;
    const rowScale = rowLength > 1 ? rowLength - 1 : 1;

    // Shift existing spectrogram content up by one pixel row.
    bufferCtx.drawImage(buffer, 0, -1);

    // Render the new row at the bottom of the buffer.
    const imageData = bufferCtx.createImageData(bufferWidth, 1);
    for (let x = 0; x < bufferWidth; x += 1) {
      const idx = Math.floor((x / widthScale) * rowScale);
      const color = palette(row[idx]);
      const offset = x * 4;

      imageData.data[offset] = color.r;
      imageData.data[offset + 1] = color.g;
      imageData.data[offset + 2] = color.b;
      imageData.data[offset + 3] = 255;
    }

    bufferCtx.putImageData(imageData, 0, bufferHeight - 1);

    // Paint the buffer to the visible canvas, scaling for device pixel ratio.
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(buffer, 0, 0, canvas.width, canvas.height);
  }, [row, rowLength]);

  return <canvas ref={canvasRef} className="plot-canvas" />;
};

export default SpectrogramCanvas;
