import json
import os

# Build notebook structure
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "cells": []
}

cells = []

# Helper functions
def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source + "\n"] if not source.endswith("\n") else [source]
    }

def code_cell(source):
    src = source if isinstance(source, list) else source.split('\n')
    # Ensure each line ends properly for the format
    formatted = []
    for i, line in enumerate(src):
        if i < len(src) - 1 or line:  # Keep lines
            if not line.endswith('\n') and i < len(src) - 1:
                formatted.append(line + '\n')
            elif line.endswith('\n'):
                formatted.append(line)
            else:
                formatted.append(line + '\n' if i < len(src) - 1 else line)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": formatted if formatted else [source]
    }

# Add all cells (truncated for brevity but I'll write a complete one)
# Actually let me write the full notebook content
pass
