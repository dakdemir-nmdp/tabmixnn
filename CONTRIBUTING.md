# Contributing to TabMixNN

Thank you for your interest in contributing to TabMixNN! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

---

## Getting Started

### Ways to Contribute

- **Bug Reports**: Found a bug? Open an issue with a minimal reproducible example
- **Feature Requests**: Have an idea? Open an issue describing the feature and use case
- **Documentation**: Improve docs, fix typos, add examples
- **Code**: Fix bugs, implement features, optimize performance
- **Examples**: Add new examples demonstrating TabMixNN capabilities

### Before You Start

1. Check existing issues to avoid duplicates
2. For major changes, open an issue first to discuss
3. Make sure you can run the existing tests

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/tabmixnn.git
cd tabmixnn
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

---

## Contributing Guidelines

### Code Style

We follow PEP 8 with some modifications:

```bash
# Format code with black
black tabmixnn/

# Check style
flake8 tabmixnn/ --max-line-length=100 --ignore=E203,W503

# Type checking (recommended)
mypy tabmixnn/
```

**Key Guidelines**:
- Line length: 100 characters max
- Use type hints where possible
- Docstrings for all public functions/classes (Google style)
- Meaningful variable names (no single letters except loop counters)

### Example Docstring

```python
def my_function(param1: str, param2: int = 5) -> bool:
    """
    Brief one-line description.

    Longer description if needed, explaining what the function does,
    when to use it, and any important details.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, optional
        Description of param2 (default: 5)

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> my_function("test", 10)
    True
    """
    pass
```

### Commit Messages

Use clear, descriptive commit messages:

```bash
# Good
git commit -m "Fix: Handle empty dataframes in fit() method"
git commit -m "Feature: Add support for Gamma distribution family"
git commit -m "Docs: Update random slopes example with real data"

# Not as good
git commit -m "fix bug"
git commit -m "update"
```

**Format**:
- Start with type: `Fix:`, `Feature:`, `Docs:`, `Test:`, `Refactor:`
- Use present tense ("Add feature" not "Added feature")
- Reference issue numbers when applicable (#123)

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=tabmixnn --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Writing Tests

- Add tests for all new features
- Add tests for bug fixes (test the fix)
- Use fixtures for reusable test data
- Name tests clearly: `test_<function>_<scenario>_<expected_result>`

**Example**:

```python
import pytest
import pandas as pd
from tabmixnn import TabMixNNRegressor

def test_fit_with_empty_dataframe_raises_error():
    """Test that fitting with empty dataframe raises ValueError."""
    model = TabMixNNRegressor(formula="y ~ x + (1|group)")
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        model.fit(df)

def test_predict_returns_correct_shape():
    """Test that predict returns array of correct shape."""
    # Setup
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [4, 5, 6],
        'group': ['A', 'A', 'B']
    })
    model = TabMixNNRegressor(formula="y ~ x + (1|group)", epochs=1)
    model.fit(df)

    # Test
    predictions = model.predict(df)

    # Assert
    assert len(predictions) == len(df)
    assert predictions.shape == (3,)
```

---

## Documentation

### Types of Documentation

1. **Code Comments**: Explain *why*, not *what*
2. **Docstrings**: API documentation for users
3. **Examples**: Practical demonstrations
4. **Vignettes**: In-depth tutorials
5. **README**: Quick start and overview

### Adding Examples

Examples should:
- Use real datasets (not synthetic)
- Be self-contained and runnable
- Include clear comments
- Show output/results
- Save outputs to `examples/output/`

### Updating Vignettes

When adding major features:
1. Update relevant vignette section
2. Add example to vignette if appropriate
3. Update the vignette README if needed

---

## Submitting Changes

### Pull Request Process

1. **Update Your Branch**
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run Tests**
   ```bash
   pytest
   black tabmixnn/
   flake8 tabmixnn/
   ```

3. **Push Changes**
   ```bash
   git push origin your-branch
   ```

4. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Fill out the PR template
   - Link any related issues

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Added tests for new features
- [ ] Updated documentation
- [ ] Added example if appropriate
- [ ] Commit messages are clear
- [ ] No merge conflicts

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
How was this tested?

## Related Issues
Closes #(issue number)

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted
```

---

## Development Workflow Example

```bash
# 1. Create branch
git checkout -b feature/add-gamma-family

# 2. Make changes
# ... edit code ...

# 3. Test
pytest tests/test_families.py

# 4. Format
black tabmixnn/families.py

# 5. Commit
git add tabmixnn/families.py tests/test_families.py
git commit -m "Feature: Add Gamma distribution family

Adds Gamma family for modeling continuous positive outcomes.
Includes tests and example usage.

Closes #42"

# 6. Push and create PR
git push origin feature/add-gamma-family
# Go to GitHub and create pull request
```

---

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Open an issue with minimal reproducible example
- **Chat**: [Add link to chat if available]

---

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

Thank you for contributing to TabMixNN!
