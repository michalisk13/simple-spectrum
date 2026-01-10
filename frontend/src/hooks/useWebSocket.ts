// React hook for connecting to the backend WebSocket stream.

import { useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import { WebSocketClient, type WebSocketConnectionState } from "../ws/client";
import type {
  ConfigAckFrame,
  ErrorFrame,
  MarkersFrame,
  SpectrogramFrame,
  SpectrumFrame,
  StatusFrame,
} from "../ws/types";

// Default WebSocket base URL for local development.
const DEFAULT_WS_BASE_URL = "ws://localhost:8000";

// Build the WebSocket stream URL using Vite environment overrides.
const buildWebSocketUrl = () => {
  // Prefer an explicit base URL from the environment, when provided.
  const envBaseUrl = import.meta.env.VITE_WS_BASE_URL as string | undefined;
  // Use the environment base or fall back to localhost.
  const baseUrl = envBaseUrl ?? DEFAULT_WS_BASE_URL;
  // Always point at the backend stream endpoint.
  return new URL("/ws/stream", baseUrl).toString();
};

// Hook return shape exposing connection state for optional UI use.
export type UseWebSocketResult = {
  // Current WebSocket connection state.
  connectionState: WebSocketConnectionState;
  // Most recent status frame, if any.
  statusFrame: StatusFrame | null;
  // Most recent spectrum payload metadata + data.
  spectrumFrame: SpectrumFrame | null;
  // Most recent spectrogram payload metadata + data.
  spectrogramFrame: SpectrogramFrame | null;
  // Most recent markers frame, if any.
  markersFrame: MarkersFrame | null;
  // Most recent config acknowledgement frame, if any.
  configAckFrame: ConfigAckFrame | null;
  // Most recent error frame, if any.
  errorFrame: ErrorFrame | null;
  // Latest spectrum payload metadata + data (mutable ref for high-rate reads).
  latestSpectrumFrameRef: MutableRefObject<SpectrumFrame | null>;
  // Latest spectrogram payload metadata + data (mutable ref for high-rate reads).
  latestSpectrogramFrameRef: MutableRefObject<SpectrogramFrame | null>;
};

export type UseWebSocketOptions = {
  // Optional callback invoked for every status frame.
  onStatusFrame?: (frame: StatusFrame) => void;
  // Optional callback invoked for markers frames.
  onMarkersFrame?: (frame: MarkersFrame) => void;
  // Optional callback invoked for config acknowledgement frames.
  onConfigAckFrame?: (frame: ConfigAckFrame) => void;
  // Optional callback invoked for error frames.
  onErrorFrame?: (frame: ErrorFrame) => void;
  // Optional callback invoked for every spectrum payload.
  onSpectrumFrame?: (frame: SpectrumFrame) => void;
  // Optional callback invoked for every spectrogram payload.
  onSpectrogramFrame?: (frame: SpectrogramFrame) => void;
};

// Connects to the backend stream and exposes frames to the UI.
export const useWebSocket = (
  options: UseWebSocketOptions = {},
): UseWebSocketResult => {
  // Track connection state for potential UI indicators.
  const [connectionState, setConnectionState] =
    useState<WebSocketConnectionState>("disconnected");
  const [statusFrame, setStatusFrame] = useState<StatusFrame | null>(null);
  const [spectrumFrame, setSpectrumFrame] = useState<SpectrumFrame | null>(null);
  const [spectrogramFrame, setSpectrogramFrame] =
    useState<SpectrogramFrame | null>(null);
  const [markersFrame, setMarkersFrame] = useState<MarkersFrame | null>(null);
  const [configAckFrame, setConfigAckFrame] =
    useState<ConfigAckFrame | null>(null);
  const [errorFrame, setErrorFrame] = useState<ErrorFrame | null>(null);
  // Persist the client instance for the lifetime of the hook.
  const clientRef = useRef<WebSocketClient | null>(null);
  const latestSpectrumFrameRef = useRef<SpectrumFrame | null>(null);
  const latestSpectrogramFrameRef = useRef<SpectrogramFrame | null>(null);
  const optionsRef = useRef(options);

  // Memoize the URL so we only build it once per hook instance.
  const wsUrl = useMemo(() => buildWebSocketUrl(), []);

  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  useEffect(() => {
    // Create a new WebSocket client with reconnect support.
    const client = new WebSocketClient(
      { url: wsUrl },
      {
        // Update state when the connection changes.
        onConnectionStateChange: (state) => {
          setConnectionState(state);
          if (state === "disconnected") {
            latestSpectrumFrameRef.current = null;
            latestSpectrogramFrameRef.current = null;
            setStatusFrame(null);
            setSpectrumFrame(null);
            setSpectrogramFrame(null);
            setMarkersFrame(null);
            setConfigAckFrame(null);
            setErrorFrame(null);
          }
        },
        // Track status frames for UI synchronization.
        onStatusFrame: (frame: StatusFrame) => {
          setStatusFrame(frame);
          optionsRef.current.onStatusFrame?.(frame);
        },
        onMarkersFrame: (frame: MarkersFrame) => {
          setMarkersFrame(frame);
          optionsRef.current.onMarkersFrame?.(frame);
        },
        onConfigAckFrame: (frame: ConfigAckFrame) => {
          setConfigAckFrame(frame);
          optionsRef.current.onConfigAckFrame?.(frame);
        },
        onErrorFrame: (frame: ErrorFrame) => {
          setErrorFrame(frame);
          optionsRef.current.onErrorFrame?.(frame);
        },
        onSpectrumFrame: (frame: SpectrumFrame) => {
          latestSpectrumFrameRef.current = frame;
          setSpectrumFrame(frame);
          optionsRef.current.onSpectrumFrame?.(frame);
        },
        onSpectrogramFrame: (frame: SpectrogramFrame) => {
          latestSpectrogramFrameRef.current = frame;
          setSpectrogramFrame(frame);
          optionsRef.current.onSpectrogramFrame?.(frame);
        },
        // Log errors to help diagnose connection issues.
        onError: (event) => {
          // Browser-level WebSocket errors are expected if the backend is down.
          console.warn("WebSocket error (backend offline or restarting)", event);
        },
      },
    );

    // Store the client for later teardown.
    clientRef.current = client;
    // Initiate the connection immediately.
    client.connect();

    return () => {
      // Close the socket when the component unmounts.
      // Note: React Strict Mode runs effects twice in dev, so quick connect/
      // disconnect cycles are normal during development.
      client.disconnect();
      clientRef.current = null;
    };
  }, [wsUrl]);

  return {
    connectionState,
    statusFrame,
    spectrumFrame,
    spectrogramFrame,
    markersFrame,
    configAckFrame,
    errorFrame,
    latestSpectrumFrameRef,
    latestSpectrogramFrameRef,
  };
};
