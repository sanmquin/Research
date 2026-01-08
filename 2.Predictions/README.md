# 2.Predictions - LLM-Driven Feature Engineering for Video Performance Prediction

This directory contains experiments on using Large Language Models (LLMs) to generate interpretable features for predicting YouTube video performance. The approach uses a three-step iterative process: **Describe**, **Rank**, and **Reflexion**.

## Overview

**Research Question:** Can LLM-generated features improve prediction accuracy for YouTube video view counts?

**Approach:** Rather than using traditional feature engineering or embeddings, we leverage LLMs to:
1. Identify predictive features from high/low performing videos
2. Score all videos on these features
3. Iteratively refine features using a Reflexion-based learning loop

## Three-Step Process

### Step 1: Describe (`1.Describe.ts`)

**Purpose:** Extract interpretable features from video titles by contrasting top and bottom performers.

**Process:**
1. **Select Videos:** Sorts all videos by view count and selects the top 20 and bottom 20 performers
2. **Contrast Analysis:** Prompts the LLM to identify 5 distinct features that differentiate high performers from low performers
3. **Feature Extraction:** LLM returns structured features with:
   - `name`: Short identifier for the feature
   - `summary`: One-sentence description
   - `description`: Detailed explanation with examples

**Key Functions:**
- `getFeatures()`: Generates initial feature set by contrasting top/bottom videos
- `addFeature()`: During Reflexion, suggests new features based on prediction errors

**Example Features Generated:**
- Specificity of topic/concept
- Emotional appeal or intrigue
- Use of numbers or concrete details
- Question-based titles
- Clickbait elements

**LLM Models Used:**
- Primary: `gemma-3-27b-it` (via Google Gemini API)
- Alternative: Ollama models for local inference

---

### Step 2: Rank (`2.Rank.ts`)

**Purpose:** Score all videos on the identified features using LLM judgement.

**Process:**
1. **Batch Processing:** Videos are processed in batches (20 for Gemini, 10 for Ollama)
2. **Feature Scoring:** For each video title, the LLM assigns a score from 0-10 for each feature
3. **Structured Output:** Returns JSON with title and feature scores: `{ title: string, features: { [featureName]: number } }`

**Key Functions:**
- `rankVideos()`: Scores all videos on given features using batch processing
- Handles retries and validation to ensure all videos are scored

**Technical Details:**
- Uses dynamic JSON schema generation based on feature names
- Normalizes titles using NFKC to handle encoding differences
- Validates responses to ensure all requested videos are scored

---

### Step 3: Reflexion (`3.Reflexion.ts`)

**Purpose:** Iteratively improve prediction accuracy through a learning loop inspired by the Reflexion paradigm.

**The Reflexion Process:**

Reflexion is a technique for improving LLM performance through self-reflection and iterative refinement. In this context, it enables the model to learn which features are most predictive and to discover new features that address prediction errors.

#### Reflexion Loop Components:

1. **Training Phase:**
   - Train multivariate linear regression on current features
   - Predict video performance on training set
   - Calculate prediction errors

2. **Reflection Phase:**
   - Identify the **least important feature** using linear model coefficients
   - Find the **20 worst predictions** (largest absolute errors)
   - Analyze patterns in over-predictions and under-predictions

3. **Learning Phase:**
   - LLM receives:
     - Current feature set
     - Worst predictions with actual vs predicted views
     - List of previously failed features (to avoid repetition)
   - LLM proposes a new feature that might explain the errors
   - Remove the least important feature
   - Add the newly proposed feature

4. **Validation Phase:**
   - Score all videos on the new feature
   - Train new model with updated feature set
   - Compare validation set performance
   - **Accept** new feature if error decreases
   - **Reject** new feature if error increases (add to failed features list)

5. **Iteration:**
   - Repeat for up to 10 iterations or until convergence
   - Track all failed features to avoid re-suggesting them

**Key Functions:**
- `featureVideos()`: Combines Steps 1 & 2 to get initial featured videos
- `trainFeatureModel()`: Trains linear regression on features
- `updateFeatures()`: Identifies and removes least important feature
- `getWorstPredictions()`: Finds largest prediction errors for reflection
- `reflexionStep()`: Executes one complete iteration of the Reflexion loop
- `runReflexion()`: Orchestrates the full iterative process

---

## Linear Model & Feature Importance

### Model Architecture

The prediction model uses **multivariate linear regression** with the following structure:

```
log(views + 1) = β₀ + β₁·recent₁ + β₂·recent₂ + ... + β₅·recent₅ 
                 + β₆·feature₁ + β₇·feature₂ + ... + βₙ·featureₙ
```

**Input Features:**
- **Recent Performance (5 features):** Log-transformed views from the previous 5 videos
- **LLM-Generated Features (5-10 features):** Scores (0-10) on identified title characteristics

**Target Variable:**
- `log(views + 1)`: Log-transformed view count (log scale linearizes the relationship)

### Internal Mechanism: Identifying Dimensions for Improvement

The linear model's coefficients (weights) reveal which features matter most for predictions. Here's how we use them to guide feature refinement:

#### 1. **Coefficient Extraction**

After training, the model contains a weight vector `β`:
```typescript
const [intercept, ...recentCoefficients, ...featureCoefficients] = mlr.weights
```

The feature coefficients `β₆, β₇, ..., βₙ` represent the impact of each LLM-generated feature on predicted log-views.

#### 2. **Importance Calculation**

Feature importance is measured by **absolute value of coefficients**:
```typescript
const importance = Math.abs(featureCoefficients[i][0])
```

**Why absolute value?**
- A large positive coefficient means the feature strongly increases predicted views
- A large negative coefficient means the feature strongly decreases predicted views
- Both are important; we care about magnitude of impact, not direction
- Small absolute values indicate the feature has minimal predictive power

#### 3. **Least Important Feature Identification**

```typescript
const leastFeatureIdx = featuresValue.reduce((minIdx, curr, idx, arr) => 
    curr.importance < arr[minIdx].importance ? idx : minIdx
, 0)
```

This finds the feature with the **smallest absolute coefficient**, indicating it contributes least to predictions.

#### 4. **Pruning Strategy**

By removing the least important feature each iteration:
- We maintain a **fixed-size feature set** (5 features)
- We make room for new, potentially better features
- We prevent overfitting by avoiding too many features
- We focus the LLM's attention on replacing weak features

#### 5. **Error-Driven Feature Discovery**

The 20 worst predictions provide evidence of **missing dimensions**:
```typescript
const worstPredictions = getWorstPredictions(videos, mlr, X, Y)
// Returns videos where |predicted - actual| is largest
```

These errors fall into two categories:
- **Under-predictions (delta < 0):** Videos that performed better than expected
  - Suggests missing positive features (e.g., unexploited appeal factors)
- **Over-predictions (delta > 0):** Videos that performed worse than expected
  - Suggests missing negative features (e.g., undetected weaknesses)

The LLM analyzes these systematic errors to propose features that would explain the discrepancies.

#### 6. **Validation & Selection**

Each proposed feature is evaluated on a held-out validation set:
```typescript
const improved = newModelResults.averageLogError < previousModelResults.averageLogError
```

Only features that **demonstrably improve** validation error are retained. This prevents the LLM from proposing spurious features that overfit the training data.

### Why This Mechanism Works

1. **Interpretability:** Linear coefficients are human-readable and show exactly how each feature affects predictions
2. **Guided Search:** Removing weak features and analyzing errors focuses the search on productive directions
3. **Regularization:** Fixed feature count prevents unbounded model complexity
4. **LLM Strength:** LLMs excel at pattern recognition in language, making them ideal for discovering linguistic features in titles
5. **Empirical Validation:** Every feature must prove its worth on unseen data

---

## Data Types

### Core Interfaces

```typescript
interface iVideo {
    id: string;
    title: string;
    views: number;
    likes: number;
    comments: number;
    date: string;
    duration: number;
}

interface iFeature {
    name: string;          // Feature identifier
    summary: string;       // Short description
    description: string;   // Detailed explanation
}

interface iTrainingVideo extends iVideo {
    recentViews: number[];  // Log-views from previous 5 videos
}

interface iFeaturedVideo extends iTrainingVideo {
    features: Record<string, number>;  // LLM scores for each feature
}

interface iPrediction {
    title: string;
    actual: number;        // Actual views
    predicted: number;     // Predicted views
    diff: number;          // Absolute difference
    delta: number;         // Log-space error (pred - actual)
}
```

---

## Usage

### Running the Full Pipeline

```bash
# Step 1: Extract features from video data
npx ts-node 2.Predictions/1.Describe.ts

# Step 2: Score videos using features
npx ts-node 2.Predictions/2.Rank.ts

# Step 3: Run Reflexion-based improvement
npx ts-node 2.Predictions/3.Reflexion.ts
```

### Programmatic Usage

```typescript
import { featureVideos, runReflexion } from './3.Reflexion'
import { getFeatures } from './1.Describe'
import { rankVideos } from './2.Rank'

// Get initial features
const features = await getFeatures(videos)

// Rank all videos
const rankings = await rankVideos(videos, features)

// Run full Reflexion loop
const { features: finalFeatures, videos: featuredVideos } = await runReflexion(videos, './data')
```

---

## Model Configuration

### Train/Validation Split
- **Training Set:** 80% of videos (randomly shuffled each iteration)
- **Validation Set:** 20% of videos (held out for unbiased evaluation)

### Hyperparameters
- **Initial Features:** 5 features identified by contrast analysis
- **Reflexion Iterations:** Up to 10 iterations
- **Feature Count:** Fixed at 5 features (prune least important before adding new)
- **Batch Size:** 20 videos for Gemini, 10 for Ollama

### Evaluation Metrics
- **Average Log Error:** Mean absolute error in log-space
- **Average Views Error:** Mean absolute error in raw view counts
- **Typical Factor:** `exp(avg_log_error)` - represents typical multiplicative error
- **Total Error:** Sum of absolute errors across all predictions

---

## Output Files

The Reflexion process generates structured output in `data/reflexion/`:

```
data/reflexion/
├── features-0.json          # Feature evolution for each iteration
├── features-1.json
├── ...
├── videos-0.json            # Rankings and training/validation split
├── results-0.json           # Model performance metrics
├── features.json            # Final features and failed features
└── reflexionVideos.json     # All videos with final feature scores
```

Each iteration saves:
- **features-N.json:** Current features, new feature proposed, feature removed, failed features list
- **videos-N.json:** Training/validation split, feature scores for all videos
- **results-N.json:** Previous model vs. new model performance comparison

---

## Technical Dependencies

```json
{
  "@google/genai": "^1.34.0",              // Google Gemini API client
  "ml-regression-multivariate-linear": "^2.0.4",  // Linear regression
  "@types/node": "^25.0.3"                 // TypeScript support
}
```

### LLM Providers
- **Gemini API:** Primary provider for large models (gemma-3-27b-it, gemma-3-12b-it)
- **Ollama:** Alternative for local inference (requires local Ollama server)

---

## Results

Detailed experimental results are available in:
- **Summary:** [Results.md](./Results.md)
- **Full Data:** [Google Spreadsheet](https://docs.google.com/spreadsheets/d/11_zT3_zUno5xC2Jrxf0cNRS2Z-TW_vevC5cjBYw6ByQ/edit?usp=sharing)

### Key Findings
- LLM-generated features provide **interpretable** predictors for video performance
- Reflexion loop successfully identifies and replaces weak features
- Linear model coefficients effectively guide feature refinement
- Typical prediction accuracy: Within 2-3x of actual views (typical factor)

---

## Related Work

This approach draws inspiration from:
- **Reflexion** (Shinn et al., 2023): Self-reflection for improving agent performance
- **Feature Engineering:** Automated discovery of predictive features
- **Interpretable ML:** Using linear models for explainable predictions
- **LLM-based Data Science:** Leveraging language models for structured reasoning about data

For comprehensive research context, methodology, and analysis, see [`Research.pdf`](../Research.pdf).

---

## Author

Santiago M.

## License

ISC
