"""
Lightweight structured logger for Railway deployment.
AWS-free implementation.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredLogger:
    """Simple structured logger without AWS dependencies."""

    def __init__(self, name: str, level: str = None):
        self.logger = logging.getLogger(name)
        log_level = (level or 'INFO').upper()
        self.logger.setLevel(getattr(logging, log_level, logging.INFO))

        # Configure formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )

        # Remove existing handlers to avoid duplication
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add console handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _format_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Format message with context."""
        data = {
            'message': message,
            'context': context or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        return json.dumps(data, default=str)

    def log_error(self, message: str, error: Exception = None,
                  context: Dict[str, Any] = None) -> None:
        """Log error with optional exception details."""
        ctx = context or {}
        if error:
            ctx['error_type'] = type(error).__name__
            ctx['error_message'] = str(error)
        self.logger.error(self._format_message(message, ctx))

    def log_warning(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log warning with context."""
        self.logger.warning(self._format_message(message, context))

    def log_info(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log info with context."""
        self.logger.info(self._format_message(message, context))

    def log_debug(self, message: str, context: Dict[str, Any] = None) -> None:
        """Log debug with context."""
        self.logger.debug(self._format_message(message, context))
