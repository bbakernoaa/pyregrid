#!/usr/bin/env python3
"""
Coverage report generation script for PyRegrid.

This script generates comprehensive coverage reports in multiple formats
and provides detailed analysis of coverage metrics.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def generate_coverage_reports(output_dir="coverage_reports"):
    """Generate comprehensive coverage reports."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Generating coverage reports in {output_path}...")
    
    # Run tests with coverage
    commands = [
        "pytest --cov=pyregrid --cov-report=term-missing",
        "pytest --cov=pyregrid --cov-report=html:htmlcov --cov-report=xml:coverage.xml --cov-report=json:coverage.json --cov-report=lcov:coverage.lcov --cov-fail-under=90 --cov-branch --cov-report=functions",
    ]
    
    for cmd in commands:
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"Command failed: {cmd}")
            print(f"Error: {stderr}")
            return False
        print(f"✓ {cmd}")
        if stdout:
            print(f"Output: {stdout}")
    
    # Generate additional analysis
    generate_coverage_analysis(output_path)
    
    print(f"\nCoverage reports generated successfully in {output_path}/")
    print("HTML report available at: htmlcov/index.html")
    print("XML report available at: coverage.xml")
    print("JSON report available at: coverage.json")
    
    return True


def generate_coverage_analysis(output_path):
    """Generate additional coverage analysis."""
    print("Generating coverage analysis...")
    
    # Parse XML coverage report
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse("coverage.xml")
        root = tree.getroot()
        
        line_rate = float(root.get('line-rate', 0))
        branch_rate = float(root.get('branch-rate', 0))
        
        analysis = {
            "line_coverage_percent": round(line_rate * 100, 2),
            "branch_coverage_percent": round(branch_rate * 100, 2),
            "total_lines": int(root.get('lines-covered', 0)) + int(root.get('lines-valid', 0)) - int(root.get('lines-covered', 0)),
            "covered_lines": int(root.get('lines-covered', 0)),
            "missing_lines": int(root.get('lines-valid', 0)) - int(root.get('lines-covered', 0)),
        }
        
        # Write analysis to file
        with open(output_path / "coverage_analysis.txt", "w") as f:
            f.write("PyRegrid Coverage Analysis\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Line Coverage: {analysis['line_coverage_percent']}%\n")
            f.write(f"Branch Coverage: {analysis['branch_coverage_percent']}%\n")
            f.write(f"Total Lines: {analysis['total_lines']}\n")
            f.write(f"Covered Lines: {analysis['covered_lines']}\n")
            f.write(f"Missing Lines: {analysis['missing_lines']}\n\n")
            
            if analysis['line_coverage_percent'] >= 90:
                f.write("✓ Coverage meets 90% threshold\n")
            else:
                f.write("✗ Coverage below 90% threshold\n")
                
            if analysis['branch_coverage_percent'] >= 80:
                f.write("✓ Branch coverage meets 80% threshold\n")
            else:
                f.write("⚠ Branch coverage below 80%\n")
        
        print(f"Coverage analysis saved to {output_path}/coverage_analysis.txt")
        
    except Exception as e:
        print(f"Error generating coverage analysis: {e}")


def check_coverage_thresholds():
    """Check if coverage meets thresholds."""
    print("Checking coverage thresholds...")
    
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse("coverage.xml")
        root = tree.getroot()
        
        line_rate = float(root.get('line-rate', 0))
        branch_rate = float(root.get('branch-rate', 0))
        
        print(f"Line coverage: {line_rate:.2%}")
        print(f"Branch coverage: {branch_rate:.2%}")
        
        if line_rate < 0.90:
            print("ERROR: Line coverage below 90% threshold")
            return False
        
        if branch_rate < 0.80:
            print("WARNING: Branch coverage below 80%")
        
        print("Coverage thresholds met!")
        return True
        
    except Exception as e:
        print(f"Error checking coverage thresholds: {e}")
        return False


def clean_coverage_files():
    """Clean up coverage files."""
    coverage_files = [
        ".coverage",
        "coverage.xml",
        "coverage.json",
        "coverage.lcov",
        "htmlcov/",
    ]
    
    print("Cleaning up coverage files...")
    
    for file_path in coverage_files:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                print(f"Removed directory: {file_path}")
            else:
                path.unlink()
                print(f"Removed file: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate coverage reports for PyRegrid")
    parser.add_argument(
        "--output-dir", 
        default="coverage_reports",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Check if coverage meets thresholds"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up coverage files"
    )
    
    args = parser.parse_args()
    
    if args.clean:
        clean_coverage_files()
        return
    
    if args.check_thresholds:
        success = check_coverage_thresholds()
        sys.exit(0 if success else 1)
    
    success = generate_coverage_reports(args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()