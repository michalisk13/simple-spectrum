// WebSocket client wrapper for the backend streaming endpoint.

import type {
  ConfigAckFrame,
  ErrorFrame,
  MarkersFrame,
  SpectrogramFrame,
  SpectrogramMetaFrame,
  SpectrumFrame,
  SpectrumMetaFrame,
  StatusFrame,
  WebSocketFrame,
} from "./types";

const SPAY_MAGIC = "SPAY";
const SPAY_HEADER_BYTES = 32;
const SPAY_VERSION = 1;
const SPAY_KIND_SPECTRUM = 1;
const SPAY_KIND_SPECTROGRAM = 2;
const textDecoder = new TextDecoder();

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
  // Invoked for each markers frame received from the server.
  onMarkersFrame?: (frame: MarkersFrame) => void;
  // Invoked for config acknowledgement frames from the server.
  onConfigAckFrame?: (frame: ConfigAckFrame) => void;
  // Invoked for error frames received from the server.
  onErrorFrame?: (frame: ErrorFrame) => void;
  // Invoked for each spectrum payload paired with metadata.
  onSpectrumFrame?: (frame: SpectrumFrame) => void;
  // Invoked for each spectrogram payload paired with metadata.
  onSpectrogramFrame?: (frame: SpectrogramFrame) => void;
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
  private pendingSpectrumMeta = new Map<string, SpectrumMetaFrame>();
  private pendingSpectrogramMeta = new Map<string, SpectrogramMetaFrame>();

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

    this.resetStreamState();
    // Reset manual close so reconnect logic can run.
    this.manualClose = false;
    // Notify consumers that we are attempting to connect.
    this.callbacks.onConnectionStateChange?.("connecting");

    // Open the WebSocket connection.
    const socket = new WebSocket(this.url);
    socket.binaryType = "arraybuffer";
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
      if (typeof event.data === "string") {
        this.handleJsonMessage(event.data);
        return;
      }

      if (event.data instanceof ArrayBuffer) {
        this.handleBinaryMessage(event.data);
        return;
      }

      if (event.data instanceof Blob) {
        void event.data.arrayBuffer().then((buffer) => {
          this.handleBinaryMessage(buffer);
        });
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
      this.resetStreamState();

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

    this.resetStreamState();
  }

  private handleJsonMessage(raw: string) {
    try {
      // Parse the JSON payload from the server.
      const frame = JSON.parse(raw) as WebSocketFrame;
      // Route status frames to the appropriate callback.
      if (frame.type === "status") {
        this.callbacks.onStatusFrame?.(frame as StatusFrame);
        return;
      }
      if (frame.type === "markers") {
        this.callbacks.onMarkersFrame?.(frame as MarkersFrame);
        return;
      }
      if (frame.type === "config_ack") {
        this.callbacks.onConfigAckFrame?.(frame as ConfigAckFrame);
        return;
      }
      if (frame.type === "error") {
        this.callbacks.onErrorFrame?.(frame as ErrorFrame);
        return;
      }
      if (frame.type === "spectrum_meta") {
        const meta = frame as SpectrumMetaFrame;
        this.pendingSpectrumMeta.set(meta.payload_id, meta);
        return;
      }
      if (frame.type === "spectrogram_meta") {
        const meta = frame as SpectrogramMetaFrame;
        this.pendingSpectrogramMeta.set(meta.payload_id, meta);
      }
    } catch (error) {
      // Log JSON parsing errors without breaking the client.
      console.warn("Failed to parse WebSocket message", error);
    }
  }

  private handleBinaryMessage(buffer: ArrayBuffer) {
    if (buffer.byteLength < SPAY_HEADER_BYTES) {
      console.warn("Received truncated binary payload");
      return;
    }

    const header = parseSpayHeader(buffer);
    if (!header) {
      return;
    }

    const payloadBuffer = buffer.slice(SPAY_HEADER_BYTES);
    if (header.kind === SPAY_KIND_SPECTRUM) {
      const meta = this.pendingSpectrumMeta.get(header.payloadId);
      if (!meta) {
        console.warn("Spectrum payload received without metadata");
        return;
      }
      const expectedBytes = header.elementCount * Float32Array.BYTES_PER_ELEMENT;
      if (payloadBuffer.byteLength < expectedBytes) {
        console.warn("Spectrum payload shorter than expected");
        return;
      }
      const payload = new Float32Array(payloadBuffer, 0, header.elementCount);
      this.pendingSpectrumMeta.delete(header.payloadId);
      this.callbacks.onSpectrumFrame?.({ meta, payload });
      return;
    }

    if (header.kind === SPAY_KIND_SPECTROGRAM) {
      const meta = this.pendingSpectrogramMeta.get(header.payloadId);
      if (!meta) {
        console.warn("Spectrogram payload received without metadata");
        return;
      }

      if (meta.dtype === "u8") {
        if (payloadBuffer.byteLength < header.elementCount) {
          console.warn("Spectrogram payload shorter than expected");
          return;
        }
        const payload = new Uint8Array(payloadBuffer, 0, header.elementCount);
        this.pendingSpectrogramMeta.delete(header.payloadId);
        this.callbacks.onSpectrogramFrame?.({ meta, payload });
        return;
      }

      const expectedBytes = header.elementCount * Float32Array.BYTES_PER_ELEMENT;
      if (payloadBuffer.byteLength < expectedBytes) {
        console.warn("Spectrogram payload shorter than expected");
        return;
      }
      const payload = new Float32Array(payloadBuffer, 0, header.elementCount);
      this.pendingSpectrogramMeta.delete(header.payloadId);
      this.callbacks.onSpectrogramFrame?.({ meta, payload });
    }
  }

  private resetStreamState() {
    this.pendingSpectrumMeta.clear();
    this.pendingSpectrogramMeta.clear();
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

type SpayHeader = {
  kind: number;
  payloadId: string;
  elementCount: number;
};

const parseSpayHeader = (buffer: ArrayBuffer): SpayHeader | null => {
  const magic = textDecoder.decode(new Uint8Array(buffer, 0, 4));
  if (magic !== SPAY_MAGIC) {
    console.warn("Invalid SPAY magic");
    return null;
  }

  const view = new DataView(buffer, 0, SPAY_HEADER_BYTES);
  const version = view.getUint16(4, true);
  if (version !== SPAY_VERSION) {
    console.warn("Unsupported SPAY version");
    return null;
  }

  const kind = view.getUint16(6, true);
  if (kind !== SPAY_KIND_SPECTRUM && kind !== SPAY_KIND_SPECTROGRAM) {
    console.warn("Unsupported SPAY kind");
    return null;
  }

  const payloadBytes = new Uint8Array(buffer, 8, 16);
  const payloadId = bytesToUuid(payloadBytes);
  const elementCount = view.getUint32(24, true);
  return { kind, payloadId, elementCount };
};

const bytesToUuid = (bytes: Uint8Array): string => {
  const hex = Array.from(bytes)
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
};
