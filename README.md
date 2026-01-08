# Research

A comparative study of LLM-based and embedding-based approaches for information retrieval and numeric prediction tasks.

## Overview

This repository contains two experimental research tracks that evaluate the effectiveness of Large Language Models (LLMs) compared to traditional embedding-based methods:

1. **Information Retrieval** - Comparing LLM reasoning vs. embeddings for ranking research papers
2. **Numeric Predictions** - Evaluating LLM-generated features for predicting YouTube video performance

For detailed research context, methodology, and theoretical background, please refer to [`Research.pdf`](./Research.pdf).

## Repository Structure

```
.
├── 1.Retrieval/          # Information Retrieval Experiments
│   ├── 1.Zero-shot.ts    # Basic LLM ranking approach
│   ├── 2.Multidimensional.ts  # Multi-factor reasoning experiment
│   ├── Results.md        # ⭐ Detailed retrieval results
│   ├── results/          # ⭐ Raw experimental data (JSON)
│   │   ├── 1.Zero-Shot/
│   │   ├── 2.Factors/
│   │   └── 3.Weights/
│   ├── eval/             # Evaluation scripts
│   ├── utils/            # Helper utilities (ranking, fetching, Gemini client)
│   └── sideLab/          # Verification tools
│
├── 2.Predictions/        # Numeric Prediction Experiments
│   ├── 1.Describe.ts     # Feature extraction from video data
│   ├── 2.Rank.ts         # Feature-based ranking
│   ├── 3.Reflexion.ts    # Reflexion-based improvement
│   ├── Results.md        # ⭐ Link to prediction results spreadsheet
│   └── utils/            # Helper utilities (Ollama, Gemma clients)
│
├── Research.pdf          # 📄 Full research paper with context and analysis
├── package.json          # Node.js dependencies
└── README.md             # This file
```

## Experiments

### 1. Information Retrieval

**Research Question:** Can LLM reasoning outperform embedding-based similarity for ranking research papers?

**Task:** Given a seed paper and a target paper, rank candidate papers (1-hop citations from the seed) by their likelihood of citing the target.

**Approaches Tested:**

1. **Zero-shot** (`1.Zero-shot.ts`) - Direct LLM ranking without additional reasoning
2. **Multi-Dimensional Reasoning** (`2.Multidimensional.ts`) - LLM evaluates multiple factors (explanations and contrastive explanations)
3. **Contrastive Inference** - Similar to multi-dimensional but with contrast emphasis
4. **Positive Weighted Factors** - LLM reasoning with weighted positive factors
5. **Negative Weighted Factors** - LLM reasoning with weighted negative factors

**Results Location:** 📊
- Summary: [`1.Retrieval/Results.md`](./1.Retrieval/Results.md)
- Raw data: [`1.Retrieval/results/`](./1.Retrieval/results/) (timestamped JSON files)

**Key Findings:**
- Zero-shot: Embeddings won 32 vs. LLM 23 (Avg rank: 11.43 vs 12.57)
- Multi-dimensional: Embeddings won 11 vs. LLM 8 (Avg rank: 11.20 vs 10.20)
- Negative Weighted: Embeddings won 21 vs. LLM 14 (Avg rank: 10.00 vs 12.35)

### 2. Numeric Predictions

**Research Question:** Can LLM-generated features improve prediction accuracy for YouTube video view counts?

**Task:** Predict video performance using features extracted and scored by LLMs.

**Workflow:**

1. **Feature Extraction** (`1.Describe.ts`) - Analyze top/bottom performing videos to identify key title features
2. **Feature Scoring** (`2.Rank.ts`) - Score all videos on extracted features
3. **Reflexion** (`3.Reflexion.ts`) - Iteratively improve predictions using reflection

**Results Location:** 📊
- Summary: [`2.Predictions/Results.md`](./2.Predictions/Results.md)
- Detailed data: [Google Spreadsheet](https://docs.google.com/spreadsheets/d/11_zT3_zUno5xC2Jrxf0cNRS2Z-TW_vevC5cjBYw6ByQ/edit?usp=sharing)

## Installation

### Prerequisites
- Node.js (v18 or higher)
- TypeScript
- API access to:
  - Google Gemini API (for Retrieval experiments)
  - Ollama or Gemma (for Prediction experiments)

### Setup

```bash
# Clone the repository
git clone https://github.com/sanmquin/Research.git
cd Research

# Install dependencies
npm install

# Configure API keys (set environment variables)
export GEMINI_API_KEY="your-gemini-api-key"
# For Ollama, ensure it's running locally on default port
```

## Usage

### Running Retrieval Experiments

```bash
# Run zero-shot retrieval
npx ts-node 1.Retrieval/1.Zero-shot.ts

# Run multi-dimensional reasoning
npx ts-node 1.Retrieval/2.Multidimensional.ts

# Run evaluation scripts
npx ts-node 1.Retrieval/eval/1.Zero-shot.ts
npx ts-node 1.Retrieval/eval/2.Multidimensional.ts
# ... etc
```

### Running Prediction Experiments

```bash
# Extract features from video data
npx ts-node 2.Predictions/1.Describe.ts

# Score videos using features
npx ts-node 2.Predictions/2.Rank.ts

# Run reflexion-based improvement
npx ts-node 2.Predictions/3.Reflexion.ts
```

## Dependencies

- **@google/genai** (^1.34.0) - Google Gemini API client for LLM inference
- **ml-regression-multivariate-linear** (^2.0.4) - Linear regression for prediction tasks
- **@types/node** (^25.0.3) - TypeScript definitions for Node.js

## Results Summary

### Information Retrieval
The experiments demonstrate varying performance between embedding-based and LLM-based approaches:
- Simple embedding similarity performs well in zero-shot scenarios
- Multi-dimensional LLM reasoning shows promise with improved average rankings
- Weighted factor approaches provide nuanced performance trade-offs

### Numeric Predictions
LLM-generated features enable interpretable prediction models for video performance, with detailed results available in the linked spreadsheet.

## Research Context

For comprehensive details on:
- Research methodology and experimental design
- Theoretical background and related work
- Detailed analysis and discussion of results
- Future directions and limitations

Please refer to **[`Research.pdf`](./Research.pdf)** in this repository.

## Author

Santiago M.

## License

ISC
