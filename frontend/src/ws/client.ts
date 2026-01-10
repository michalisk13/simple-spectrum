// WebSocket client wrapper for the backend streaming endpoint.

import type { StatusFrame, WebSocketFrame } from "./types";

// Connection state values exposed to consumers of the client.
export type WebSocketConnectionState =
  | "connecting"
  | "connected"
  | "disconnected";

// Callback signatures used by the WebSocket client.
type WebSocketCallbacks = {
  // Invoked when the connection state changes.
  onConnectionStateChange?: (state: WebSocketConnectionState) => void;
  // Invoked for each status frame received from the server.
  onStatusFrame?: (frame: StatusFrame) => void;
  // Invoked when an error occurs on the WebSocket.
  onError?: (event: Event) => void;
};

// Configuration options for the WebSocket client.
type WebSocketClientOptions = {
  // Absolute WebSocket URL to connect to.
  url: string;
  // Reconnect delay in milliseconds, before applying backoff.
  reconnectDelayMs?: number;
  // Maximum reconnect delay in milliseconds.
  maxReconnectDelayMs?: number;
};

// Lightweight WebSocket client with reconnect support.
export class WebSocketClient {
  private readonly url: string;
  private readonly callbacks: WebSocketCallbacks;
  private readonly reconnectDelayMs: number;
  private readonly maxReconnectDelayMs: number;
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
  private manualClose = false;

  constructor(options: WebSocketClientOptions, callbacks: WebSocketCallbacks) {
    // Persist the target URL for each reconnect attempt.
    this.url = options.url;
    // Store callbacks for notifying the UI.
    this.callbacks = callbacks;
    // Store reconnect tuning values.
    this.reconnectDelayMs = options.reconnectDelayMs ?? 1000;
    this.maxReconnectDelayMs = options.maxReconnectDelayMs ?? 10000;
  }

  // Connect to the WebSocket endpoint and begin listening for frames.
  connect() {
    // Avoid duplicate connections if a socket already exists.
    if (this.socket) {
      return;
    }

    // Reset manual close so reconnect logic can run.
    this.manualClose = false;
    // Notify consumers that we are attempting to connect.
    this.callbacks.onConnectionStateChange?.("connecting");

    // Open the WebSocket connection.
    const socket = new WebSocket(this.url);
    this.socket = socket;

    // Handle successful connection.
    socket.addEventListener("open", () => {
      // Reset reconnect attempts on success.
      this.reconnectAttempts = 0;
      // Notify consumers that we are connected.
      this.callbacks.onConnectionStateChange?.("connected");
    });

    // Handle incoming messages.
    socket.addEventListener("message", (event) => {
      // Only parse string frames for now (binary handled later).
      if (typeof event.data !== "string") {
        return;
      }

      try {
        // Parse the JSON payload from the server.
        const frame = JSON.parse(event.data) as WebSocketFrame;
        // Route status frames to the appropriate callback.
        if (frame.frame_type === "status") {
          this.callbacks.onStatusFrame?.(frame as StatusFrame);
        }
      } catch (error) {
        // Log JSON parsing errors without breaking the client.
        console.warn("Failed to parse WebSocket message", error);
      }
    });

    // Handle socket errors.
    socket.addEventListener("error", (event) => {
      this.callbacks.onError?.(event);
    });

    // Handle socket closure and trigger reconnect if needed.
    socket.addEventListener("close", () => {
      // Clean up the socket reference.
      this.socket = null;
      // Notify consumers that we are disconnected.
      this.callbacks.onConnectionStateChange?.("disconnected");

      // Exit early if the close was user-initiated.
      if (this.manualClose) {
        return;
      }

      // Schedule a reconnect attempt.
      this.scheduleReconnect();
    });
  }

  // Close the WebSocket connection and stop reconnect attempts.
  disconnect() {
    // Mark the close as manual to prevent reconnects.
    this.manualClose = true;

    // Clear any pending reconnect timers.
    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }

    // Close the active socket if it exists.
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  // Compute a backoff delay for reconnecting.
  private getReconnectDelay(): number {
    // Grow the delay linearly with the attempt count.
    const delay = this.reconnectDelayMs * (this.reconnectAttempts + 1);
    // Cap the delay to avoid excessive waits.
    return Math.min(delay, this.maxReconnectDelayMs);
  }

  // Schedule a reconnect attempt using the configured backoff.
  private scheduleReconnect() {
    // Avoid multiple reconnect timers.
    if (this.reconnectTimeoutId) {
      return;
    }

    // Increment attempts and compute delay.
    this.reconnectAttempts += 1;
    const delayMs = this.getReconnectDelay();

    // Schedule a reconnect and clear the timeout once fired.
    this.reconnectTimeoutId = setTimeout(() => {
      this.reconnectTimeoutId = null;
      this.connect();
    }, delayMs);
  }
}
