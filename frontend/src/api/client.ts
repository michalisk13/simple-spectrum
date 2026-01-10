// API client wrapper for the REST endpoints used by the frontend.

import type { ApiError, ApiStatusResponse } from "./types";
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

  // Load the current connection status from the backend.
  async getStatus(): Promise<ApiStatusResponse | null> {
    try {
      const response = await this.request<ApiStatusResponse>("/status");

      // Surface engine-reported errors through notifications.
      if (response.error) {
        this.onError({
          message: response.error.message,
          details: response.error.error_code,
        });
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
}
