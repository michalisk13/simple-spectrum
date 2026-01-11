import { useEffect, useRef, type MutableRefObject } from "react";
import { useAnimationFrame } from "../../hooks/useAnimationFrame";
import type { SpectrogramFrame } from "../../ws/types";

const palette = (value: number) => {
  const clamped = Math.min(1, Math.max(0, value));
  const r = Math.floor(20 + clamped * 80);
  const g = Math.floor(40 + clamped * 140);
  const b = Math.floor(120 + clamped * 120);

  return { r, g, b };
};

export type SpectrogramCanvasProps = {
  frameRef: MutableRefObject<SpectrogramFrame | null>;
};

const SpectrogramCanvas = ({ frameRef }: SpectrogramCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bufferRef = useRef<HTMLCanvasElement | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const lastSeqRef = useRef<number | null>(null);
  const lastFrameRef = useRef<SpectrogramFrame | null>(null);
  const imageRowRef = useRef<ImageData | null>(null);
  const needsRedrawRef = useRef(false);

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
      imageRowRef.current = null;
      needsRedrawRef.current = true;
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

  useAnimationFrame(() => {
    const canvas = canvasRef.current;
    const buffer = bufferRef.current;
    if (!canvas || !buffer) {
      return;
    }

    const bufferCtx = buffer.getContext("2d");
    const ctx = canvas.getContext("2d");
    if (!bufferCtx || !ctx) {
      return;
    }

    const frame = frameRef.current;
    if (frame) {
      lastFrameRef.current = frame;
    }

    const lastFrame = lastFrameRef.current;
    if (!lastFrame) {
      if (lastSeqRef.current !== null || needsRedrawRef.current) {
        lastSeqRef.current = null;
        needsRedrawRef.current = false;
        bufferCtx.fillStyle = "#0b1220";
        bufferCtx.fillRect(0, 0, buffer.width, buffer.height);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(buffer, 0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const seq = lastFrame.meta.seq;
    if (seq === lastSeqRef.current && !needsRedrawRef.current) {
      return;
    }
    lastSeqRef.current = seq;
    needsRedrawRef.current = false;

    const row = lastFrame.payload;
    const rowLength = row.length;
    if (!rowLength) {
      return;
    }

    let scaleMin = 0;
    let scaleRange = 0;
    if (lastFrame.meta.dtype === "f32") {
      const range = lastFrame.meta.db_max - lastFrame.meta.db_min;
      if (Number.isFinite(range) && range > 0) {
        scaleMin = lastFrame.meta.db_min;
        scaleRange = range;
      }
    }

    const bufferWidth = buffer.width;
    const bufferHeight = buffer.height;
    const widthScale = bufferWidth > 1 ? bufferWidth - 1 : 1;
    const rowScale = rowLength > 1 ? rowLength - 1 : 1;

    // Shift existing spectrogram content up by one pixel row.
    bufferCtx.drawImage(buffer, 0, -1);

    // Render the new row at the bottom of the buffer.
    let imageData = imageRowRef.current;
    if (!imageData || imageData.width !== bufferWidth) {
      imageData = bufferCtx.createImageData(bufferWidth, 1);
      imageRowRef.current = imageData;
    }
    for (let x = 0; x < bufferWidth; x += 1) {
      const idx = Math.floor((x / widthScale) * rowScale);
      const raw = row[idx];
      const normalized =
        row instanceof Uint8Array
          ? raw / 255
          : scaleRange > 0
            ? (raw - scaleMin) / scaleRange
            : 0;
      const color = palette(normalized);
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
  }, 30);

  return <canvas ref={canvasRef} className="plot-canvas" />;
};

export default SpectrogramCanvas;
