// Shared helpers for user-facing notifications.

import { notifications } from "@mantine/notifications";
import { IconAlertTriangle } from "@tabler/icons-react";
import type { ApiError } from "../../api/types";

// Display a non-blocking notification for API failures.
export const notifyApiError = (error: ApiError): void => {
  const message = error.details
    ? `${error.message}: ${error.details}`
    : error.message;

  notifications.show({
    title: "API error",
    message,
    color: "red",
    icon: <IconAlertTriangle size={18} />,
    autoClose: 8000,
  });
};
