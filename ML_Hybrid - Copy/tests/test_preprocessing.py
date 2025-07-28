"""
Tests for text preprocessing module.
"""

import pytest
from unittest.mock import patch, Mock
import spacy

from preprocessing.text_processor import TextProcessor


class TestTextProcessor:
    """Test cases for TextProcessor class."""

    def test_text_processor_initialization(self, mock_config):
        """Test TextProcessor initialization."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()
            assert processor is not None
            assert hasattr(processor, "clean_text")
            assert hasattr(processor, "process_summaries")

    def test_clean_text_basic(self, mock_config):
        """Test basic text cleaning functionality."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test HTML removal
            html_text = "<p>This is a <b>test</b> text</p>"
            cleaned = processor.clean_text(html_text)
            assert "<p>" not in cleaned
            assert "<b>" not in cleaned
            assert "test" in cleaned

            # Test case normalization
            mixed_case = "This Is A Test TEXT"
            cleaned = processor.clean_text(mixed_case)
            assert cleaned == "this is a test text"

    def test_clean_text_advanced(self, mock_config):
        """Test advanced text cleaning features."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test with special characters and encoding issues
            problematic_text = "This text has special chars: éñü & symbols: @#$%"
            cleaned = processor.clean_text(problematic_text)
            assert len(cleaned) > 0
            assert "text" in cleaned.lower()

    def test_detect_language(self, mock_config):
        """Test language detection functionality."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test English detection
            english_text = "This is an English text"
            lang = processor.detect_language(english_text)
            assert lang == "en"

            # Test with short text
            short_text = "Hello"
            lang = processor.detect_language(short_text)
            assert lang in ["en", "unknown"]

    def test_lemmatize_text(self, mock_config):
        """Test lemmatization functionality."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test lemmatization
            text = "running quickly through the fields"
            lemmatized = processor.lemmatize_text(text)
            assert "run" in lemmatized or "running" in lemmatized
            assert len(lemmatized) > 0

    def test_tokenize_sentences(self, mock_config):
        """Test sentence tokenization."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test sentence splitting
            text = "This is sentence one. This is sentence two. And this is sentence three."
            sentences = processor.tokenize_sentences(text)
            assert len(sentences) == 3
            assert all(len(sentence) > 0 for sentence in sentences)

    def test_validate_text(self, mock_config):
        """Test text validation."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test valid text
            valid_text = "This is a valid text with sufficient length"
            assert processor.validate_text(valid_text) == True

            # Test too short text
            short_text = "Short"
            assert processor.validate_text(short_text) == False

            # Test empty text
            empty_text = ""
            assert processor.validate_text(empty_text) == False

    def test_process_summaries(self, sample_texts, mock_config):
        """Test processing multiple summaries."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Process summaries
            results = processor.process_summaries(sample_texts)

            # Check results structure
            assert isinstance(results, list)
            assert len(results) == len(sample_texts)

            # Check each result has required fields
            for result in results:
                assert "original" in result
                assert "processed" in result
                assert "valid" in result
                assert "language" in result
                assert "length" in result

    def test_process_summaries_with_invalid_texts(self, mock_config):
        """Test processing with invalid texts."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Mix of valid and invalid texts
            texts = [
                "Valid text with sufficient length",
                "",  # Empty
                "Short",  # Too short
                "Another valid text that should be processed correctly",
            ]

            results = processor.process_summaries(texts)

            # Check that invalid texts are marked as such
            assert results[0]["valid"] == True
            assert results[1]["valid"] == False
            assert results[2]["valid"] == False
            assert results[3]["valid"] == True

    def test_remove_boilerplate(self, mock_config):
        """Test boilerplate removal."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test with common boilerplate
            boilerplate_text = (
                "Dear Sir/Madam, This is the main content. Best regards, John"
            )
            cleaned = processor.clean_text(boilerplate_text)

            # Should remove common boilerplate phrases
            assert "Dear Sir/Madam" not in cleaned
            assert "Best regards" not in cleaned
            assert "main content" in cleaned.lower()

    def test_handle_unicode_issues(self, mock_config):
        """Test handling of Unicode issues."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test with Unicode issues
            unicode_text = "This text has unicode issues: café résumé naïve"
            cleaned = processor.clean_text(unicode_text)

            # Should handle Unicode gracefully
            assert len(cleaned) > 0
            assert "text" in cleaned.lower()

    def test_generate_preprocessing_statistics(self, sample_texts, mock_config):
        """Test preprocessing statistics generation."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Process summaries
            results = processor.process_summaries(sample_texts)

            # Generate statistics
            stats = processor.generate_statistics(results)

            # Check statistics structure
            assert "total_summaries" in stats
            assert "processed_count" in stats
            assert "valid_count" in stats
            assert "invalid_count" in stats
            assert "language_distribution" in stats
            assert "length_statistics" in stats

            # Check basic statistics
            assert stats["total_summaries"] == len(sample_texts)
            assert stats["processed_count"] == len(sample_texts)
            assert stats["valid_count"] + stats["invalid_count"] == len(sample_texts)

    def test_error_handling(self, mock_config):
        """Test error handling in text processing."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Test with None input
            with pytest.raises(ValueError):
                processor.clean_text(None)

            # Test with non-string input
            with pytest.raises(ValueError):
                processor.clean_text(123)

            # Test with very long text
            long_text = "A" * 10000
            result = processor.clean_text(long_text)
            assert len(result) <= mock_config["preprocessing"]["max_summary_length"]

    def test_spacy_model_loading(self, mock_config):
        """Test spaCy model loading."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Check if spaCy model is loaded
            assert hasattr(processor, "nlp")
            assert processor.nlp is not None

    def test_nltk_data_download(self, mock_config):
        """Test NLTK data download."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            # This should not raise an error even if NLTK data is not downloaded
            processor = TextProcessor()
            assert processor is not None
