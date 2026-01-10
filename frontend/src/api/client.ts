// API client wrapper for the REST endpoints used by the frontend.

import type {
  ApiCommandResponse,
  ApiConfigResponse,
  ApiError,
  ApiPresetsResponse,
  ApiStatusResponse,
  ApplyPresetRequest,
  ApplyPresetResponse,
  ConfigUpdatePayload,
  EngineError,
  SdrCommandRequest,
} from "./types";
import { notifyApiError } from "../components/notifications/notify";

// Configuration for the API client instance.
type ApiClientOptions = {
  // Base URL for API requests, defaulting to the dev proxy path.
  baseUrl?: string;
  // Optional error handler override for API failures.
  onError?: (error: ApiError) => void;
};

// Defaults to the local proxy path configured in Vite.
const DEFAULT_BASE_URL = "/api";

// Runtime guard for verifying a value already matches ApiError.
const isApiError = (value: unknown): value is ApiError => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  return typeof record.message === "string";
};

// Normalize any thrown value into a consistent ApiError shape.
const toApiError = (error: unknown): ApiError => {
  if (isApiError(error)) {
    return error;
  }
  if (error instanceof Error) {
    return { message: error.message };
  }
  return { message: "Unexpected API error" };
};

// Typed REST client that wraps the fetch API.
export class ApiClient {
  private baseUrl: string;
  private onError: (error: ApiError) => void;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? DEFAULT_BASE_URL;
    this.onError = options.onError ?? notifyApiError;
  }

  // Load the current configuration from the backend.
  async getConfig(): Promise<ApiConfigResponse | null> {
    try {
      return await this.request<ApiConfigResponse>("/config");
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Apply partial configuration updates.
  async updateConfig(payload: ConfigUpdatePayload): Promise<ApiConfigResponse | null> {
    try {
      return await this.request<ApiConfigResponse>("/config", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Load the current connection status from the backend.
  async getStatus(): Promise<ApiStatusResponse | null> {
    try {
      const response = await this.request<ApiStatusResponse>("/status");

      // Surface engine-reported errors through notifications.
      this.handleEngineError(response.error);

      return response;
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Connect to the SDR with an optional URI override.
  async connectSdr(uri?: string): Promise<ApiCommandResponse | null> {
    try {
      const payload: SdrCommandRequest | undefined = uri ? { uri } : undefined;
      const response = await this.request<ApiCommandResponse>("/sdr/connect", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: payload ? JSON.stringify(payload) : undefined,
      });
      this.handleCommandResponse(response);
      return response;
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Disconnect from the SDR.
  async disconnectSdr(): Promise<ApiCommandResponse | null> {
    try {
      const response = await this.request<ApiCommandResponse>("/sdr/disconnect", {
        method: "POST",
      });
      this.handleCommandResponse(response);
      return response;
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Disconnect and reconnect the SDR.
  async reconnectSdr(): Promise<ApiCommandResponse | null> {
    try {
      const response = await this.request<ApiCommandResponse>("/sdr/reconnect", {
        method: "POST",
      });
      this.handleCommandResponse(response);
      return response;
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Test connectivity to the SDR without maintaining a connection.
  async testSdr(uri?: string): Promise<ApiCommandResponse | null> {
    try {
      const payload: SdrCommandRequest | undefined = uri ? { uri } : undefined;
      const response = await this.request<ApiCommandResponse>("/sdr/test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: payload ? JSON.stringify(payload) : undefined,
      });
      this.handleCommandResponse(response);
      return response;
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // List available presets.
  async listPresets(): Promise<ApiPresetsResponse | null> {
    try {
      return await this.request<ApiPresetsResponse>("/presets");
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Apply a named preset.
  async applyPreset(payload: ApplyPresetRequest): Promise<ApplyPresetResponse | null> {
    try {
      const response = await this.request<ApplyPresetResponse>("/presets/apply", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        this.onError({ message: "Preset could not be applied." });
      }
      return response;
    } catch (error) {
      this.onError(toApiError(error));
      return null;
    }
  }

  // Core fetch wrapper that parses JSON and raises friendly errors.
  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...init?.headers,
      },
    });

    if (!response.ok) {
      const details = await response.text();
      throw {
        message: `Request failed with status ${response.status}`,
        status: response.status,
        details: details || undefined,
      } satisfies ApiError;
    }

    return (await response.json()) as T;
  }

  private handleCommandResponse(response: ApiCommandResponse): void {
    if (response.error) {
      this.handleEngineError(response.error);
      return;
    }
    if (!response.ok) {
      this.onError({ message: "Command failed to execute." });
    }
  }

  private handleEngineError(error: EngineError | null | undefined): void {
    if (!error) {
      return;
    }
    this.onError({
      message: error.message,
      details: error.error_code,
    });
  }
}
