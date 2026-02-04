# pygatos

Python library for GATOS (Generative AI-enabled Theme Organization and Structuring) - a framework for automated qualitative coding and thematic analysis using large language models.

## Features

- **Codebook Generation**: Automatically generate qualitative codebooks from text data using LLM-powered analysis
- **Code Application**: Apply existing codebooks to new text data
- **Multiple LLM Backends**: Support for Ollama (local) and Cerebras (cloud) inference
- **Thematic Analysis Pipeline**: End-to-end pipeline from raw text to organized themes
- **Visualization**: Cluster visualizations, frequency distributions, and growth trends
- **Flexible I/O**: Load data from CSV, JSON, and text files; export codebooks in multiple formats

## Installation

```bash
pip install pygatos
```

With optional dependencies:

```bash
# Visualization support
pip install pygatos[viz]

# Cerebras cloud backend
pip install pygatos[cerebras]

# Development tools
pip install pygatos[dev]
```

## Quick Start

```python
from pygatos import GATOSConfig, GATOSPipeline

config = GATOSConfig(
    llm_backend="ollama",
    model_name="llama3",
)

pipeline = GATOSPipeline(config)
codebook = pipeline.run(texts)
```

## CLI

```bash
pygatos --help
```

## License

MIT
