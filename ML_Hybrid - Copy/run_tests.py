#!/usr/bin/env python3
"""
Test runner script for ML Hybrid Theme Analysis system.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", coverage=True, verbose=True):
    """
    Run tests with specified options.

    Args:
        test_type: Type of tests to run (all, unit, integration, api)
        coverage: Whether to run with coverage
        verbose: Whether to run in verbose mode
    """

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test type filters
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "api":
        cmd.extend(["-m", "api"])
    elif test_type == "clustering":
        cmd.extend(["-m", "clustering"])
    elif test_type == "preprocessing":
        cmd.extend(["-m", "preprocessing"])
    elif test_type == "utils":
        cmd.extend(["-m", "utils"])

    # Add coverage options
    if coverage:
        cmd.extend(
            ["--cov=src", "--cov-report=term-missing", "--cov-report=html:htmlcov"]
        )

    # Add verbose option
    if verbose:
        cmd.append("-v")

    # Add test directory
    cmd.append("tests/")

    print(f"Running tests: {' '.join(cmd)}")

    # Run tests
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Tests completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False


def run_specific_test(test_file):
    """
    Run a specific test file.

    Args:
        test_file: Path to the test file
    """
    cmd = ["python", "-m", "pytest", test_file, "-v"]

    print(f"Running specific test: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Test completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with exit code {e.returncode}")
        return False


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(
        description="Run tests for ML Hybrid Theme Analysis system"
    )
    parser.add_argument(
        "--type",
        choices=[
            "all",
            "unit",
            "integration",
            "api",
            "clustering",
            "preprocessing",
            "utils",
        ],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Run tests without coverage"
    )
    parser.add_argument("--quiet", action="store_true", help="Run tests in quiet mode")
    parser.add_argument("--file", help="Run a specific test file")

    args = parser.parse_args()

    if args.file:
        success = run_specific_test(args.file)
    else:
        success = run_tests(
            test_type=args.type, coverage=not args.no_coverage, verbose=not args.quiet
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
