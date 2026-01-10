import { useEffect, useRef } from "react";

// Animation frame hook with an optional FPS cap for predictable pacing.
export const useAnimationFrame = (
  callback: (deltaMs: number) => void,
  fps = 60,
) => {
  // Track the active animation frame ID so we can cancel on cleanup.
  const frameIdRef = useRef<number | null>(null);
  // Store timestamps to throttle updates to the requested FPS.
  const lastTimeRef = useRef<number | null>(null);
  const minFrameTime = 1000 / fps;

  useEffect(() => {
    const tick = (time: number) => {
      const lastTime = lastTimeRef.current ?? time;
      const delta = time - lastTime;

      // Only invoke the callback when we've hit the frame budget.
      if (delta >= minFrameTime) {
        lastTimeRef.current = time;
        callback(delta);
      }

      frameIdRef.current = window.requestAnimationFrame(tick);
    };

    frameIdRef.current = window.requestAnimationFrame(tick);

    return () => {
      if (frameIdRef.current !== null) {
        window.cancelAnimationFrame(frameIdRef.current);
      }
      frameIdRef.current = null;
      lastTimeRef.current = null;
    };
  }, [callback, minFrameTime]);
};
