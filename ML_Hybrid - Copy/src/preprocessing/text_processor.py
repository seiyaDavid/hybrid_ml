"""
Text preprocessing module for ML Hybrid Theme Analysis.

This module provides comprehensive text preprocessing capabilities including
cleaning, normalization, tokenization, and feature extraction.
"""

import re
import string
import ftfy
from typing import List, Dict, Any, Optional
from loguru import logger
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

log = logger


class TextProcessor:
    """
    Comprehensive text preprocessing pipeline for theme analysis.

    This class provides extensive text cleaning, normalization, and
    feature extraction capabilities optimized for theme discovery.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text processor with configuration.

        Args:
            config: Configuration dictionary containing preprocessing settings
        """
        self.config = config
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

        # Add custom stop words
        custom_stops = config.get("preprocessing", {}).get("custom_stop_words", [])
        self.stop_words.update(custom_stops)

        log.info("TextProcessor initialized successfully")

    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning and normalization.

        Args:
            text: Raw text input

        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""

        # Fix encoding issues
        text = ftfy.fix_text(text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove special characters but keep punctuation for sentence boundaries
        text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\(\)]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the text.

        Args:
            text: Input text

        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            return detect(text)
        except LangDetectException:
            log.warning(f"Could not detect language for text: {text[:100]}...")
            return "en"  # Default to English

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of words
        """
        words = word_tokenize(text.lower())
        return [word for word in words if word.isalnum()]

    def remove_stop_words(self, words: List[str]) -> List[str]:
        """
        Remove stop words from tokenized text.

        Args:
            words: List of words

        Returns:
            List of words with stop words removed
        """
        return [word for word in words if word.lower() not in self.stop_words]

    def lemmatize_words(self, words: List[str]) -> List[str]:
        """
        Lemmatize words to their base form.

        Args:
            words: List of words

        Returns:
            List of lemmatized words
        """
        return [self.lemmatizer.lemmatize(word) for word in words]

    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate text patterns.

        Args:
            text: Input text

        Returns:
            Text with boilerplate removed
        """
        # Common boilerplate patterns
        boilerplate_patterns = [
            r"^\s*(issue|bug|problem|error|defect)\s*[:#]\s*",
            r"^\s*(summary|description|details)\s*[:#]\s*",
            r"^\s*(steps|reproduction|steps to reproduce)\s*[:#]\s*",
            r"^\s*(expected|actual|result)\s*[:#]\s*",
            r"^\s*(environment|version|browser|os)\s*[:#]\s*",
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text.strip()

    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """
        Extract key phrases from text using simple heuristics.

        Args:
            text: Input text
            max_phrases: Maximum number of phrases to extract

        Returns:
            List of key phrases
        """
        # Simple key phrase extraction based on frequency and position
        words = self.tokenize_words(text)
        words = self.remove_stop_words(words)
        words = self.lemmatize_words(words)

        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top phrases
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_phrases]]

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Complete text preprocessing pipeline.

        Args:
            text: Raw text input

        Returns:
            Dictionary containing processed text and metadata
        """
        if not text:
            return {
                "cleaned_text": "",
                "sentences": [],
                "words": [],
                "key_phrases": [],
                "language": "en",
                "word_count": 0,
                "sentence_count": 0,
            }

        # Clean text
        cleaned_text = self.clean_text(text)

        # Detect language
        language = self.detect_language(cleaned_text)

        # Remove boilerplate
        cleaned_text = self.remove_boilerplate(cleaned_text)

        # Tokenize sentences
        sentences = self.tokenize_sentences(cleaned_text)

        # Tokenize words
        words = self.tokenize_words(cleaned_text)
        words = self.remove_stop_words(words)
        words = self.lemmatize_words(words)

        # Extract key phrases
        key_phrases = self.extract_key_phrases(cleaned_text)

        return {
            "cleaned_text": cleaned_text,
            "sentences": sentences,
            "words": words,
            "key_phrases": key_phrases,
            "language": language,
            "word_count": len(words),
            "sentence_count": len(sentences),
        }

    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw texts

        Returns:
            List of preprocessing results
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.preprocess_text(text)
                results.append(result)
                log.debug(f"Processed text {i+1}/{len(texts)}")
            except Exception as e:
                log.error(f"Error processing text {i+1}: {str(e)}")
                results.append(
                    {
                        "cleaned_text": "",
                        "sentences": [],
                        "words": [],
                        "key_phrases": [],
                        "language": "en",
                        "word_count": 0,
                        "sentence_count": 0,
                    }
                )

        return results

    def get_text_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the text corpus.

        Args:
            texts: List of raw texts

        Returns:
            Dictionary containing text statistics
        """
        processed_texts = self.preprocess_batch(texts)

        total_words = sum(result["word_count"] for result in processed_texts)
        total_sentences = sum(result["sentence_count"] for result in processed_texts)
        languages = [result["language"] for result in processed_texts]

        # Language distribution
        language_dist = {}
        for lang in languages:
            language_dist[lang] = language_dist.get(lang, 0) + 1

        # Average text length
        avg_words = total_words / len(texts) if texts else 0
        avg_sentences = total_sentences / len(texts) if texts else 0

        return {
            "total_texts": len(texts),
            "total_words": total_words,
            "total_sentences": total_sentences,
            "avg_words_per_text": avg_words,
            "avg_sentences_per_text": avg_sentences,
            "language_distribution": language_dist,
            "processed_texts": processed_texts,
        }
