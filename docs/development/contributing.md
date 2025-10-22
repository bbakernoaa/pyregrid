# Contributing

We welcome contributions to PyRegrid! This document provides guidelines and information for contributing to the project.

## How to Contribute

There are several ways you can contribute to PyRegrid:

- Reporting bugs and suggesting features
- Improving documentation
- Contributing code fixes and features
- Writing tutorials and examples
- Answering questions in the community

## Development Setup

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/pyregrid/pyregrid
cd pyregrid

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all optional dependencies
pip install -e ".[dev]"
```

## Code Style

PyRegrid follows standard Python coding conventions:

- Code style: PEP 8
- Type hints: All public functions should be typed
- Documentation: Use NumPy-style docstrings
- Testing: All code should include appropriate tests

## Documentation Contributions

We particularly welcome documentation contributions:

### Adding Examples
- Create clear, self-contained examples
- Include explanations of the approach
- Follow the existing examples structure

### Improving API Documentation
- Ensure all public functions have clear docstrings
- Include usage examples where appropriate
- Document parameters and return values

### Writing Tutorials
- Focus on practical use cases
- Include complete, runnable code
- Explain the reasoning behind choices

## Testing

All contributions must include appropriate tests:

```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=pyregrid --cov-report=html

# Run specific tests
pytest tests/test_core.py
```

### YAML Configuration Testing

The documentation tests verify that `mkdocs.yml` is valid YAML. Note that:

- We use `yaml.Loader` instead of `yaml.safe_load()` because MkDocs Material theme requires Python-specific YAML tags for certain extensions
- These tags (like `!!python/name:`) are necessary for features like emoji support
- The test ensures the configuration is parseable while supporting these required tags

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation as needed
6. Submit a pull request

## Documentation-Specific Guidelines

When contributing to documentation:

- Use clear, concise language
- Provide practical examples
- Follow the existing structure
- Test examples to ensure they work
- Consider the target audience (beginners to advanced users)

## Questions?

If you have questions about contributing, feel free to open an issue or reach out to the maintainers.