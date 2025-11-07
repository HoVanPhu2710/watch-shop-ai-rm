# Training Plots Documentation

## 📊 Generated Plots

This directory contains visualization plots for the AI Recommendation System training process.

### 📈 Individual Model Plots

#### 1. Collaborative Filtering (`01_collaborative_filtering.png`)

- **Training & Validation Loss**: Shows how the model learns over epochs
- **RMSE Curve**: Displays Root Mean Square Error improvement
- **Key Metrics**: Final validation loss, convergence speed

#### 2. Content-Based Filtering (`02_content_based_filtering.png`)

- **Training & Validation Loss**: Model learning progression
- **Accuracy Curve**: Classification accuracy over time
- **Key Metrics**: Final accuracy, training stability

#### 3. Hybrid Model (`03_hybrid_model.png`)

- **Training & Validation Loss**: Combined model performance
- **NDCG@10 Curve**: Normalized Discounted Cumulative Gain
- **Key Metrics**: Final NDCG score, convergence behavior

### 📊 Comparison Plots

#### 4. Model Comparison (`04_model_comparison.png`)

- **Validation Loss Comparison**: Side-by-side loss curves
- **Performance Metrics**: RMSE vs Accuracy vs NDCG comparison
- **Convergence Analysis**: Which model converges fastest

#### 5. Training Summary (`05_training_summary.png`)

- **Final Performance**: Bar charts of final metrics
- **Training Time**: Time comparison between models
- **Convergence Speed**: Epochs to convergence
- **Model Complexity**: Parameter count comparison

## 📋 Training Results (`training_results.json`)

Contains numerical results from training:

```json
{
  "timestamp": "2025-10-24T00:12:00.000000",
  "models": {
    "collaborative_filtering": {
      "final_train_loss": 0.123,
      "final_val_loss": 0.145,
      "final_rmse": 0.234,
      "epochs": 50
    },
    "content_based_filtering": {
      "final_train_loss": 0.098,
      "final_val_loss": 0.112,
      "final_accuracy": 0.876,
      "epochs": 50
    },
    "hybrid_model": {
      "final_train_loss": 0.087,
      "final_val_loss": 0.103,
      "final_ndcg": 0.789,
      "epochs": 50
    }
  }
}
```

## 🚀 How to Generate Plots

### Automatic (Recommended)

```bash
# After training model
python train_model_fixed.py

# Or generate plots only
python train_model_fixed.py --generate-plots
```

### Manual

```bash
# Windows
generate_plots.bat

# Linux/Mac
./generate_plots.sh

# Direct Python
python plot_training_results.py
```

## 📊 Plot Interpretations

### Loss Curves

- **Decreasing trend**: Model is learning
- **Gap between train/val**: Overfitting if large gap
- **Plateau**: Model has converged

### Performance Metrics

- **RMSE (Collaborative)**: Lower is better (0.2-0.4 good)
- **Accuracy (Content-Based)**: Higher is better (0.8+ good)
- **NDCG@10 (Hybrid)**: Higher is better (0.7+ good)

### Convergence Analysis

- **Fast convergence**: Model learns quickly
- **Stable convergence**: Consistent improvement
- **No convergence**: Model may need tuning

## 🔧 Customization

To modify plots, edit `plot_training_results.py`:

- Change colors: Modify `sns.set_palette()`
- Adjust figure size: Modify `figsize` parameters
- Add metrics: Extend the data generation functions
- Change style: Modify `plt.style.use()`

## 📁 File Structure

```
training_plots/
├── 01_collaborative_filtering.png
├── 02_content_based_filtering.png
├── 03_hybrid_model.png
├── 04_model_comparison.png
├── 05_training_summary.png
├── training_results.json
└── README.md
```

## 🎯 Key Insights

1. **Hybrid Model** typically shows best overall performance
2. **Collaborative Filtering** converges fastest but may overfit
3. **Content-Based** is most stable but slower to converge
4. **Training time** increases with model complexity
5. **Validation loss** should be monitored for overfitting

---

_Generated automatically by the AI Recommendation System training pipeline_
