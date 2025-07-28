"""
Tests for utility modules (config, logger).
"""

import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from pathlib import Path

from utils.config import ConfigManager
from utils.logger import setup_logging, get_logger


class TestConfigManager:
    """Test cases for ConfigManager class."""

    def test_config_manager_initialization(self, mock_config):
        """Test ConfigManager initialization."""
        with patch("utils.config.ConfigManager._load_config", return_value=mock_config):
            config_manager = ConfigManager()
            assert config_manager.config == mock_config

    def test_get_method(self, mock_config):
        """Test ConfigManager.get method."""
        with patch("utils.config.ConfigManager._load_config", return_value=mock_config):
            config_manager = ConfigManager()

            # Test getting nested values
            assert config_manager.get("aws.region") == "us-east-1"
            assert config_manager.get("clustering.umap.n_neighbors") == 15

            # Test getting with default
            assert config_manager.get("nonexistent.key", "default") == "default"

    def test_get_aws_config(self, mock_config):
        """Test getting AWS configuration."""
        with patch("utils.config.ConfigManager._load_config", return_value=mock_config):
            config_manager = ConfigManager()
            aws_config = config_manager.get_aws_config()

            assert aws_config["region"] == "us-east-1"
            assert (
                aws_config["bedrock"]["model_id"]
                == "anthropic.claude-3-sonnet-20240229-v1:0"
            )
            assert (
                aws_config["bedrock"]["embedding_model_id"]
                == "amazon.titan-embed-text-v1"
            )

    def test_get_clustering_config(self, mock_config):
        """Test getting clustering configuration."""
        with patch("utils.config.ConfigManager._load_config", return_value=mock_config):
            config_manager = ConfigManager()
            clustering_config = config_manager.get_clustering_config()

            assert clustering_config["umap"]["n_neighbors"] == 15
            assert clustering_config["hdbscan"]["min_cluster_size"] == 5
            assert clustering_config["hyperparameter_tuning"]["enabled"] == False

    def test_environment_variable_substitution(self):
        """Test environment variable substitution."""
        test_config = {
            "aws": {
                "credentials": {
                    "access_key_id": "${AWS_ACCESS_KEY_ID}",
                    "secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
                }
            }
        }

        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"},
        ):
            with patch(
                "utils.config.ConfigManager._load_config", return_value=test_config
            ):
                config_manager = ConfigManager()

                aws_config = config_manager.get_aws_config()
                assert aws_config["credentials"]["access_key_id"] == "test_key"
                assert aws_config["credentials"]["secret_access_key"] == "test_secret"

    def test_load_config_file_not_found(self):
        """Test handling of missing config file."""
        with patch(
            "builtins.open", side_effect=FileNotFoundError("Config file not found")
        ):
            with pytest.raises(FileNotFoundError):
                ConfigManager()

    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML in config file."""
        invalid_yaml = "invalid: yaml: content: ["

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            with pytest.raises(Exception):  # Should raise YAML parsing error
                ConfigManager()


class TestLogger:
    """Test cases for logger module."""

    def test_setup_logging(self):
        """Test logging setup."""
        # Should not raise any exceptions
        setup_logging()

    def test_get_logger(self):
        """Test getting logger instance."""
        logger = get_logger("test_module")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_logger_functionality(self):
        """Test basic logger functionality."""
        logger = get_logger("test_logger")

        # Test that logging methods exist and don't raise errors
        logger.info("Test info message")
        logger.error("Test error message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")

    def test_logger_with_custom_config(self, temp_dir):
        """Test logger with custom configuration."""
        custom_config = {
            "logging": {
                "level": "DEBUG",
                "file": str(Path(temp_dir) / "test.log"),
                "rotation": "1 MB",
                "retention": "1 day",
            }
        }

        with patch(
            "utils.config.ConfigManager._load_config", return_value=custom_config
        ):
            setup_logging()
            logger = get_logger("custom_test")
            logger.info("Custom config test")

            # Check if log file was created
            log_file = Path(temp_dir) / "test.log"
            assert log_file.exists()


class TestIntegration:
    """Integration tests for utility modules."""

    def test_config_and_logger_integration(self, mock_config):
        """Test integration between config and logger."""
        with patch("utils.config.ConfigManager._load_config", return_value=mock_config):
            # Setup logging with config
            setup_logging()
            logger = get_logger("integration_test")

            # Use config manager
            config_manager = ConfigManager()
            aws_config = config_manager.get_aws_config()

            # Log config information
            logger.info(f"AWS Region: {aws_config['region']}")

            assert aws_config["region"] == "us-east-1"

    def test_error_handling(self):
        """Test error handling in utility modules."""
        # Test config manager with invalid path
        with pytest.raises(FileNotFoundError):
            ConfigManager(config_path="nonexistent_file.yaml")

        # Test logger with invalid config
        invalid_config = {"logging": {"level": "INVALID_LEVEL"}}
        with patch(
            "utils.config.ConfigManager._load_config", return_value=invalid_config
        ):
            # Should handle invalid log level gracefully
            setup_logging()
            logger = get_logger("error_test")
            logger.info("Error handling test")
