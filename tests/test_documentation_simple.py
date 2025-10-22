"""
Simple tests for documentation validation.
"""

import os
import pytest


def test_mkdocs_configuration_exists():
    """Test that mkdocs configuration file exists."""
    assert os.path.exists("mkdocs.yml"), "mkdocs.yml configuration file should exist"


def test_documentation_directories_exist():
    """Test that essential documentation directories exist."""
    expected_dirs = [
        "docs/",
        "docs/api-reference/",
        "docs/user-guide/",
        "docs/tutorials/",
        "docs/examples/",
        "docs/development/"
    ]
    
    for dir_path in expected_dirs:
        assert os.path.exists(dir_path), f"Documentation directory {dir_path} should exist"


def test_api_reference_files_exist():
    """Test that key API reference files exist."""
    api_files = [
        "docs/api-reference/pyregrid.md",
        "docs/api-reference/pyregrid.core.md",
        "docs/api-reference/pyregrid.interpolation.md"
    ]
    
    for file_path in api_files:
        assert os.path.exists(file_path), f"API reference file {file_path} should exist"


def test_notebook_integration():
    """Test that Jupyter notebook is properly integrated."""
    notebook_path = "docs/examples/notebooks/basic_regridding.ipynb"
    assert os.path.exists(notebook_path), "Jupyter notebook should exist"
    
    # Check that the notebook is referenced in mkdocs.yml
    with open("mkdocs.yml", "r") as f:
        mkdocs_config = f.read()
    
    assert "basic_regridding.ipynb" in mkdocs_config, "Notebook should be referenced in mkdocs.yml"


def test_github_workflow_exists():
    """Test that GitHub Actions workflow for documentation exists."""
    workflow_path = ".github/workflows/docs.yml"
    assert os.path.exists(workflow_path), "Documentation workflow file should exist"


def test_yaml_config_validity():
    """Test that mkdocs.yml is valid YAML."""
    import yaml
    
    try:
        with open("mkdocs.yml", "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        assert config is not None, "mkdocs.yml should be parseable"
        assert isinstance(config, dict), "mkdocs.yml should contain a dictionary"
    except Exception as e:
        pytest.fail(f"Failed to parse mkdocs.yml: {e}")
