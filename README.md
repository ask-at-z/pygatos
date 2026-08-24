# pygatos

Python library for **GATOS** (Generative AI-enabled Theme Organization and Structuring) - a method for generating qualitative codebooks from text data using LLMs and NLP techniques.

## Overview

pygatos implements an inductive approach to qualitative codebook generation, combining:
- Text embedding and clustering
- LLM-powered code suggestion
- Two-stage novelty evaluation
- Theme organization

## Installation

```bash
# From the pygatos directory
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With Cerebras cloud API support
pip install -e ".[cerebras]"

# With development tools
pip install -e ".[dev]"
```

## Quick Start

```python
from pygatos import GATOSConfig
from pygatos.core import Embedder, Clusterer, Codebook
from pygatos.llm import OllamaBackend

# Initialize components
config = GATOSConfig()
embedder = Embedder.from_config(config.embedding)
clusterer = Clusterer.from_config(config.clustering)
llm = OllamaBackend.from_config(config.llm)

# Embed texts
texts = ["Your text data here...", ...]
embeddings = embedder.embed(texts, show_progress=True)

# Cluster similar texts
labels = clusterer.fit(embeddings)
clusters = clusterer.get_cluster_contents(labels, texts)

# Process clusters to generate codes...
```

## Requirements

- Python 3.10+
- Ollama (for local LLM inference) OR Cerebras API key (for cloud inference)
- CUDA or MPS for GPU acceleration (optional)

## Configuration

The `GATOSConfig` class provides comprehensive configuration:

```python
from pygatos import GATOSConfig

config = GATOSConfig()

# Customize embedding
config.embedding.model_name = "Qwen/Qwen3-Embedding-0.6B"
config.embedding.device = "cuda"

# Customize LLM
config.llm.model = "qwen3:30b-a3b-instruct-2507-q4_K_M"
config.llm.temperature = 0.7

# Customize clustering
config.clustering.method = "agglomerative"
config.clustering.n_clusters = 50

# Customize novelty evaluation
config.novelty.similarity_threshold = 0.8
config.novelty.top_k_rag = 5

# Consolidation policy (Step 7). The default, "reject-unless-distinct", is the published
# GATOS behavior: reject a candidate code that is a specific instance of a broader code.
# Multi-seed factorial evaluation showed that policy is the pipeline's dominant recall
# bottleneck; the validated repair below changes only the instruction and moves
# ground-truth recall by +0.25 to +0.27. Cost: the codebook grows 3-8x, so audit
# duplication with pygatos.diagnostics (below).
config.novelty.policy = "keep-unless-duplicate"
```

## LLM Backends

pygatos supports multiple LLM backends:

### Ollama (Local Inference - Default)

```python
from pygatos import GATOSConfig

config = GATOSConfig()
config.llm.backend = "ollama"
config.llm.model = "qwen3:30b-a3b-instruct-2507-q4_K_M"
config.llm.base_url = "http://localhost:11434"

# Create the LLM backend
llm = config.llm.create_backend()
```

### Cerebras (Cloud API)

For fast cloud inference using Cerebras:

```python
import os
from pygatos import GATOSConfig

# Set your API key (or use CEREBRAS_API_KEY environment variable)
os.environ["CEREBRAS_API_KEY"] = "your-api-key"

config = GATOSConfig()
config.llm.backend = "cerebras"
config.llm.model = "gpt-oss-120b"  # or "llama-3.1-8b", "qwen-3-32b"

# Create the LLM backend
llm = config.llm.create_backend()
```

Available Cerebras models:
- `gpt-oss-120b` - seems to work well

## Diagnostics

`pygatos.diagnostics.compute_duplicate_rate` measures the judged true-duplicate rate of a
codebook — a parsimony check that needs **no ground truth**, which makes it the deployment
diagnostic for the `keep-unless-duplicate` policy (its known failure mode is duplication on
weak generator models). Use a judge model **different from the generator**; a local Ollama
judge keeps confidential corpora on-machine.

```python
from pygatos.config import LLMConfig
from pygatos.core import Embedder
from pygatos.diagnostics import compute_duplicate_rate

judge = LLMConfig(model="gpt-oss:120b").create_backend()  # != generator model
report = compute_duplicate_rate(
    codebook.accepted_codes, judge, Embedder(config.embedding),
    generator_model=config.llm.model,
)
print(report["dupe_rate"])  # single-judge rule; per-pair labels in report["pairs"]
```

Every CLI run also writes `manifest.json` (effective consolidation settings, prompt hashes,
environment/corpus/model provenance) and `llm_calls.jsonl` (every prompt/response sent to the
model; disable with `--no-llm-log`). Both inherit the confidentiality of the corpus.

## Architecture

```
pygatos/
├── core/           # Core building blocks
│   ├── embedder    # Text embedding
│   ├── clusterer   # Clustering algorithms
│   └── codebook    # Codebook data structures
├── generation/     # Codebook generation
├── application/    # Codebook application
├── llm/            # LLM backends
├── cache/          # Caching utilities
└── visualization/  # Visualization tools
```

## License

MIT
