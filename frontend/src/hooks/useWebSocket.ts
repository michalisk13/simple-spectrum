// React hook for connecting to the backend WebSocket stream.

import { useEffect, useMemo, useRef, useState } from "react";
import { WebSocketClient, type WebSocketConnectionState } from "../ws/client";
import type { StatusFrame } from "../ws/types";

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
};

// Connects to the backend stream and logs status frames for validation.
export const useWebSocket = (): UseWebSocketResult => {
  // Track connection state for potential UI indicators.
  const [connectionState, setConnectionState] =
    useState<WebSocketConnectionState>("disconnected");
  // Persist the client instance for the lifetime of the hook.
  const clientRef = useRef<WebSocketClient | null>(null);

  // Memoize the URL so we only build it once per hook instance.
  const wsUrl = useMemo(() => buildWebSocketUrl(), []);

  useEffect(() => {
    // Create a new WebSocket client with reconnect support.
    const client = new WebSocketClient(
      { url: wsUrl },
      {
        // Update state when the connection changes.
        onConnectionStateChange: (state) => {
          setConnectionState(state);
        },
        // Log status frames for validation until rendering is implemented.
        onStatusFrame: (frame: StatusFrame) => {
          console.info("Received StatusFrame", frame);
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

  return { connectionState };
};
