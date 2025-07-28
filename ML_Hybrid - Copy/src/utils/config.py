"""
Configuration management utilities for the ML Hybrid Theme Analysis system.

This module provides functionality to load and manage application configuration
from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ConfigManager:
    """
    Manages application configuration loading and validation.

    This class handles loading configuration from YAML files, environment
    variable substitution, and provides type-safe access to configuration
    values.
    """

    def __init__(self, config_path: str = "config/credentials.yaml"):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file with environment variable substitution."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

            with open(self.config_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Substitute environment variables
            content = self._substitute_env_vars(content)

            self._config = yaml.safe_load(content)
            logger.info(f"Configuration loaded successfully from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.

        Args:
            content: Configuration file content

        Returns:
            Content with environment variables substituted
        """
        import re

        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replace_var, content)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        keys = key.split(".")
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration section."""
        return self.get("aws", {})

    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration section."""
        return self.get("app", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.get("logging", {})

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration section."""
        return self.get("preprocessing", {})

    def get_clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration section."""
        return self.get("clustering", {})

    def get_vector_store_config(self) -> Dict[str, Any]:
        """Get vector store configuration section."""
        return self.get("vector_store", {})

    def get_theme_analysis_config(self) -> Dict[str, Any]:
        """Get theme analysis configuration section."""
        return self.get("theme_analysis", {})

    def validate_config(self) -> bool:
        """
        Validate required configuration sections.

        Returns:
            True if configuration is valid
        """
        required_sections = ["aws", "app", "logging"]

        for section in required_sections:
            if not self.get(section):
                logger.error(f"Missing required configuration section: {section}")
                return False

        # Validate AWS credentials
        aws_config = self.get_aws_config()
        if not aws_config.get("credentials", {}).get("access_key_id"):
            logger.warning("AWS access key not configured")

        logger.info("Configuration validation completed")
        return True


# Global configuration instance
config = ConfigManager()
