# Getting Started with Integral Shapley Values

Welcome to the Integral Shapley Values (ISV) toolkit! This guide will help you get started with computing Shapley values using efficient integral-based methods.

## Quick Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional: Install as package**:
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Basic Usage Example

Run the included example script to see the toolkit in action:

```bash
python example_usage.py
```

This will demonstrate:
- Different integration methods (trapezoid, Gaussian quadrature, adaptive)
- Comparison with traditional Monte Carlo
- Multiple utility functions
- Performance comparisons

### 2. Command Line Usage

Compute Shapley values directly from the command line:

```bash
# Basic usage with trapezoidal integration
python src/core/integral_shapley.py --dataset iris --utility acc --method trapezoid --num_t_samples 30 --num_MC 50

# High-precision Gaussian quadrature
python src/core/integral_shapley.py --dataset wine --utility rkhs --method gaussian --num_nodes 32

# Adaptive sampling
python src/core/integral_shapley.py --dataset cancer --utility kl --method adaptive --tolerance 1e-4

# Compute for all data points
python src/core/integral_shapley.py --dataset iris --utility acc --method trapezoid --all_points
```

### 3. Python API Usage

```python
from src.core.integral_shapley import compute_integral_shapley_value
from src.utils.utilities import utility_acc
from src.utils.model_utils import return_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
data = load_iris()
X, y = data.data, data.target
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)

# Train models
final_model = return_model('LinearSVC')
final_model.fit(x_train, y_train)
clf = return_model('LinearSVC')

# Compute Shapley value for data point 0
shapley_value = compute_integral_shapley_value(
    x_train, y_train, x_valid, y_valid, 
    i=0, clf=clf, final_model=final_model,
    utility_func=utility_acc,
    method='trapezoid', num_t_samples=30, num_MC=50
)

print(f"Shapley value: {shapley_value:.6f}")
```

## Research Experiments

Run comprehensive research experiments to study the properties of integral Shapley methods:

### 1. Efficiency Study

Compare computational efficiency across different methods:

```bash
python src/experiments/efficiency_study.py
```

This generates:
- Execution time comparisons
- Scalability analysis
- Sample efficiency metrics
- Visualizations in `results/plots/efficiency_study.png`

### 2. Smoothness Analysis

Study the smoothness properties of the integrand:

```bash
python src/experiments/smoothness_analysis.py
```

This analyzes:
- Integrand curve shapes
- Smoothness metrics
- Optimal sampling strategies
- Results saved in `results/plots/smoothness_*.png`

### 3. Convergence Study

Analyze convergence rates of different integration methods:

```bash
python src/experiments/convergence_study.py
```

This studies:
- Convergence rates vs sample size
- Error estimation
- Method comparisons
- Outputs in `results/plots/convergence_*.png`

## Testing

Run the test suite to verify functionality:

```bash
python tests/test_basic_functionality.py
```

This tests:
- Core algorithms
- Utility functions
- Edge cases
- Method consistency

## Available Methods

### Integration Methods

1. **Trapezoid** (`method='trapezoid'`):
   - Simple and robust
   - Parameters: `num_t_samples`, `num_MC`
   - Best for: General purpose, stable results

2. **Gaussian Quadrature** (`method='gaussian'`):
   - High precision
   - Parameters: `num_nodes`, `num_MC`
   - Best for: Smooth integrands, high accuracy needs

3. **Adaptive** (`method='adaptive'`):
   - Automatic sampling optimization
   - Parameters: `tolerance`, `num_MC`
   - Best for: Unknown smoothness, optimal efficiency

4. **Monte Carlo** (`method='monte_carlo'`):
   - Traditional approach for comparison
   - Parameters: `num_samples`
   - Best for: Baseline comparison

### Utility Functions

1. **Accuracy** (`utility_acc`):
   - Direct accuracy comparison
   - Fast and interpretable
   - Range: [0, 1]

2. **RKHS Similarity** (`utility_RKHS`):
   - Model similarity in reproducing kernel Hilbert space
   - Good for SVMs
   - Range: [0, 1]

3. **KL Divergence** (`utility_KL`):
   - Prediction probability similarity
   - Captures distributional differences
   - Range: [0, 1]

4. **Cosine Similarity** (`utility_cosine`):
   - Parameter vector similarity
   - Good for linear models
   - Range: [-1, 1]

## Tips for Optimal Performance

### Method Selection

- **For smooth utility functions**: Use Gaussian quadrature with fewer nodes
- **For rough/noisy utilities**: Use trapezoidal method with more samples
- **For unknown smoothness**: Start with adaptive method
- **For baseline comparison**: Use traditional Monte Carlo

### Parameter Tuning

- **num_t_samples**: Start with 20-50, increase if results are unstable
- **num_MC**: Start with 50-100, increase for more precision
- **num_nodes**: Start with 16-32 for Gaussian quadrature
- **tolerance**: Use 1e-3 to 1e-4 for adaptive method

### Performance Optimization

- Use parallel processing with `--processes` parameter
- Start with small sample sizes to test, then scale up
- Monitor convergence using the convergence study script
- Consider data preprocessing and standardization

## Output Files

Results are automatically saved in structured formats:

- **Pickle files**: `results/pickles/*.pkl` - Complete results with metadata
- **CSV files**: `results/csvs/*.csv` - Tabular data for analysis  
- **Plots**: `results/plots/*.png` - Visualizations and charts

## Common Issues and Solutions

### 1. Slow Performance
- Reduce `num_MC` or `num_t_samples`
- Use faster utility function (accuracy vs RKHS)
- Enable parallel processing
- Consider adaptive method for automatic optimization

### 2. Unstable Results
- Increase `num_MC` for more stable estimates
- Use trapezoidal method instead of Gaussian
- Check data preprocessing (standardization)
- Verify model training stability

### 3. Memory Issues
- Process data points in batches
- Reduce dataset size for testing
- Use simpler models (Linear SVC vs RBF SVC)

### 4. Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path includes src directory
- Use absolute imports or install package with `pip install -e .`

## Next Steps

1. **Explore the examples**: Run `example_usage.py` to see all features
2. **Run experiments**: Use the research scripts to understand method properties
3. **Customize for your data**: Adapt the utility functions for your specific use case
4. **Compare methods**: Use the efficiency study to find optimal parameters
5. **Contribute**: Add new integration methods or utility functions

For more detailed information, see the full README.md and the research papers referenced therein.

Happy computing! 🚀