"""
LLM-based theme analysis module for the ML Hybrid Theme Analysis system.

This module provides functionality to analyze themes using AWS Bedrock
Claude Sonnet through LangChain, including theme labeling, descriptions,
and zero-shot classification.
"""

from typing import List, Dict, Any, Optional
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from loguru import logger
import json

from ..utils.config import config
from ..utils.logger import get_logger

# Initialize logger
log = get_logger(__name__)


class ThemeAnalyzer:
    """
    Analyzes themes using AWS Bedrock Claude Sonnet through LangChain.

    This class provides functionality to generate theme names, descriptions,
    and perform zero-shot classification using Claude Sonnet.
    """

    def __init__(self):
        """Initialize the theme analyzer with AWS Bedrock configuration."""
        self.config = config.get_aws_config()
        self.model_id = self.config.get("bedrock", {}).get(
            "model_id", "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        self.theme_config = config.get_theme_analysis_config()

        try:
            # Initialize Bedrock LLM
            self.llm = BedrockChat(
                model_id=self.model_id,
                region_name=self.config.get("region", "us-east-1"),
                model_kwargs={"max_tokens": 1000, "temperature": 0.3, "top_p": 0.9},
            )

            # Initialize chains
            self._setup_chains()

            log.info(f"ThemeAnalyzer initialized with model: {self.model_id}")

        except Exception as e:
            log.error(f"Failed to initialize theme analyzer: {e}")
            raise

    def _setup_chains(self):
        """Setup LangChain chains for different analysis tasks."""

        # Theme naming chain
        theme_naming_prompt = PromptTemplate(
            input_variables=["samples"],
            template="""
You are an expert at analyzing issue summaries and identifying common themes. 
Given the following sample summaries from a cluster, generate a concise, descriptive theme name (2-4 words) that captures the main issue or topic.

Sample summaries:
{samples}

Theme name:""",
        )
        self.theme_naming_chain = LLMChain(llm=self.llm, prompt=theme_naming_prompt)

        # Theme description chain
        theme_description_prompt = PromptTemplate(
            input_variables=["theme_name", "samples"],
            template="""
You are an expert at analyzing issue summaries and providing detailed descriptions of themes.
Given the theme name "{theme_name}" and the following sample summaries, provide a detailed description of what this theme represents.

Sample summaries:
{samples}

Theme description:""",
        )
        self.theme_description_chain = LLMChain(
            llm=self.llm, prompt=theme_description_prompt
        )

        # Zero-shot classification chain
        classification_prompt = PromptTemplate(
            input_variables=["summary", "question"],
            template="""
You are an expert at analyzing issue summaries and determining if they indicate specific problems.

Question: {question}

Summary: {summary}

Answer with only "Yes" or "No":""",
        )
        self.classification_chain = LLMChain(llm=self.llm, prompt=classification_prompt)

        # Theme comparison chain
        comparison_prompt = PromptTemplate(
            input_variables=[
                "theme1_name",
                "theme1_samples",
                "theme2_name",
                "theme2_samples",
            ],
            template="""
You are an expert at comparing themes from issue summaries.

Theme 1: {theme1_name}
Theme 1 samples:
{theme1_samples}

Theme 2: {theme2_name}
Theme 2 samples:
{theme2_samples}

Provide a brief comparison of these themes, highlighting similarities and differences:""",
        )
        self.comparison_chain = LLMChain(llm=self.llm, prompt=comparison_prompt)

    def generate_theme_name(self, samples: List[str]) -> str:
        """
        Generate a theme name from sample summaries.

        Args:
            samples: List of sample summaries from a cluster

        Returns:
            Generated theme name
        """
        try:
            # Limit samples to avoid token limits
            max_samples = self.theme_config.get("max_samples_per_cluster", 10)
            limited_samples = samples[:max_samples]

            samples_text = "\n".join([f"- {sample}" for sample in limited_samples])

            response = self.theme_naming_chain.run(samples=samples_text)

            # Clean up response
            theme_name = response.strip().replace('"', "").replace("'", "")

            log.info(f"Generated theme name: {theme_name}")
            return theme_name

        except Exception as e:
            log.error(f"Error generating theme name: {e}")
            return "Unknown Theme"

    def generate_theme_description(self, theme_name: str, samples: List[str]) -> str:
        """
        Generate a detailed description for a theme.

        Args:
            theme_name: Name of the theme
            samples: List of sample summaries from the theme

        Returns:
            Generated theme description
        """
        try:
            # Limit samples to avoid token limits
            max_samples = self.theme_config.get("max_samples_per_cluster", 10)
            limited_samples = samples[:max_samples]

            samples_text = "\n".join([f"- {sample}" for sample in limited_samples])

            response = self.theme_description_chain.run(
                theme_name=theme_name, samples=samples_text
            )

            log.info(f"Generated description for theme: {theme_name}")
            return response.strip()

        except Exception as e:
            log.error(f"Error generating theme description: {e}")
            return f"Description not available for {theme_name}"

    def classify_summary(self, summary: str, question: str) -> Dict[str, Any]:
        """
        Perform zero-shot classification on a summary.

        Args:
            summary: Summary text to classify
            question: Classification question (e.g., "Does this summary indicate a data quality issue?")

        Returns:
            Dictionary containing classification result
        """
        try:
            response = self.classification_chain.run(summary=summary, question=question)

            # Parse response
            response_clean = response.strip().lower()
            is_positive = response_clean in ["yes", "true", "1"]

            result = {
                "summary": summary,
                "question": question,
                "response": response.strip(),
                "is_positive": is_positive,
                "confidence": "high" if response_clean in ["yes", "no"] else "medium",
            }

            log.info(f"Classification result: {result}")
            return result

        except Exception as e:
            log.error(f"Error classifying summary: {e}")
            return {
                "summary": summary,
                "question": question,
                "response": "Error",
                "is_positive": False,
                "confidence": "low",
            }

    def classify_data_quality_issues(self, summaries: List[str]) -> Dict[str, Any]:
        """
        Classify summaries for data quality issues using zero-shot classification with Claude Sonnet.

        Args:
            summaries: List of summaries to classify

        Returns:
            Dictionary containing classification results
        """
        log.info(
            f"Classifying {len(summaries)} summaries for data quality issues using zero-shot classification"
        )

        # Zero-shot classification question for Claude Sonnet
        question = "Does this summary indicate a data quality issue? Consider issues like data corruption, missing data, invalid data formats, duplicate entries, inconsistent data, or data integrity problems. Answer with only 'Yes' or 'No'."

        results = []
        data_quality_count = 0

        for summary in summaries:
            result = self.classify_summary(summary, question)
            results.append(result)

            if result["is_positive"]:
                data_quality_count += 1

        return {
            "results": results,
            "total_summaries": len(summaries),
            "data_quality_count": data_quality_count,
            "data_quality_percentage": (
                (data_quality_count / len(summaries)) * 100 if summaries else 0
            ),
        }

    def compare_themes(self, theme1: Dict[str, Any], theme2: Dict[str, Any]) -> str:
        """
        Compare two themes and provide analysis.

        Args:
            theme1: First theme dictionary with name and samples
            theme2: Second theme dictionary with name and samples

        Returns:
            Comparison analysis text
        """
        try:
            theme1_samples = "\n".join(
                [f"- {sample}" for sample in theme1["samples"][:5]]
            )
            theme2_samples = "\n".join(
                [f"- {sample}" for sample in theme2["samples"][:5]]
            )

            response = self.comparison_chain.run(
                theme1_name=theme1["name"],
                theme1_samples=theme1_samples,
                theme2_name=theme2["name"],
                theme2_samples=theme2_samples,
            )

            log.info(
                f"Generated comparison between {theme1['name']} and {theme2['name']}"
            )
            return response.strip()

        except Exception as e:
            log.error(f"Error comparing themes: {e}")
            return f"Comparison not available between {theme1['name']} and {theme2['name']}"

    def analyze_cluster_themes(
        self, cluster_samples: Dict[int, List[str]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Analyze themes for all clusters.

        Args:
            cluster_samples: Dictionary mapping cluster IDs to sample texts

        Returns:
            Dictionary mapping cluster IDs to theme analysis
        """
        log.info(f"Analyzing themes for {len(cluster_samples)} clusters")

        theme_analysis = {}

        for cluster_id, samples in cluster_samples.items():
            try:
                # Generate theme name
                theme_name = self.generate_theme_name(samples)

                # Generate theme description
                theme_description = self.generate_theme_description(theme_name, samples)

                # Classify for data quality issues
                classification_results = self.classify_data_quality_issues(samples)

                theme_analysis[cluster_id] = {
                    "name": theme_name,
                    "description": theme_description,
                    "sample_count": len(samples),
                    "data_quality_issues": classification_results["data_quality_count"],
                    "data_quality_percentage": classification_results[
                        "data_quality_percentage"
                    ],
                    "samples": samples[:5],  # Keep first 5 samples
                }

                log.info(f"Analyzed cluster {cluster_id}: {theme_name}")

            except Exception as e:
                log.error(f"Error analyzing cluster {cluster_id}: {e}")
                theme_analysis[cluster_id] = {
                    "name": "Unknown Theme",
                    "description": "Analysis failed",
                    "sample_count": len(samples),
                    "data_quality_issues": 0,
                    "data_quality_percentage": 0,
                    "samples": samples[:5],
                }

        return theme_analysis

    def get_theme_summary(self, theme_analysis: Dict[int, Dict[str, Any]]) -> str:
        """
        Generate a summary of all themes.

        Args:
            theme_analysis: Dictionary containing theme analysis results

        Returns:
            Formatted theme summary
        """
        summary = f"""
Theme Analysis Summary
=====================

Total Themes: {len(theme_analysis)}

"""

        for cluster_id, analysis in theme_analysis.items():
            summary += f"""
Theme {cluster_id}: {analysis['name']}
- Description: {analysis['description']}
- Sample Count: {analysis['sample_count']}
- Data Quality Issues: {analysis['data_quality_issues']} ({analysis['data_quality_percentage']:.1f}%)
"""

        return summary
