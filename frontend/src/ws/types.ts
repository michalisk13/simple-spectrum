// Typed representations of WebSocket frames consumed by the frontend.

import type { EngineStatus } from "../api/types";

// Shared base fields sent with every WebSocket frame.
export type WebSocketFrameBase = {
  // Frame discriminator used by the backend protocol contract.
  frame_type: string;
  // Monotonic timestamp in nanoseconds for ordering on the client.
  ts_monotonic_ns: number;
  // Per-session sequence number for this frame.
  seq: number;
  // Unique session identifier for this WebSocket connection.
  session_id: string;
};

// Status frame sent immediately on connect and periodically afterward.
export type StatusFrame = WebSocketFrameBase &
  EngineStatus & {
    // Explicitly narrowed discriminator for status frames.
    frame_type: "status";
  };

// Union of known WebSocket frames for type-safe handling.
export type WebSocketFrame = StatusFrame;
