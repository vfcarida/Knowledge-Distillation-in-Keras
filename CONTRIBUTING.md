# Contributing to Knowledge Distillation in Keras

First off, thank you for considering contributing to this project! It's people like you that make Knowledge Distillation in Keras such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check if there's already an issue open for it. If not, go ahead and open a new issue!

## Setting up your environment

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/Knowledge-Distillation-in-Keras.git
    cd Knowledge-Distillation-in-Keras
    ```
3.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  **Install development dependencies**:
    ```bash
    pip install -e ".[dev]"
    ```

## Development Workflow

1.  **Create a branch** for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
2.  **Make your changes**. Ensure that your code adheres to standard Python styling (PEP 8) and includes appropriate docstrings and type hints.
3.  **Run the tests**. Ensure all existing and new tests pass:
    ```bash
    pytest tests/
    ```
4.  **Commit your changes** with a descriptive commit message.
5.  **Push your branch** to your fork.
6.  **Open a Pull Request** against the main repository.

## Code Style Guide

-   Use **Type Hinting** for all function signatures.
-   Include **Docstrings** (Google or NumPy style) for all modules, classes, and public functions.
-   Follow **SOLID principles** and Clean Code practices. Avoid dumping logic in global scopes or huge monolithic functions.

Thank you for your contribution!
