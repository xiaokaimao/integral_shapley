# Integral Shapley Values (ISV)

A research-focused toolkit for computing Shapley values using efficient integral-based methods, leveraging smoothness properties for high-performance Monte Carlo approximations.

## Overview

This project implements Shapley value computation using the integral formulation:

$$\mathcal{SV}_i = \int_0^1 \mathbb{E}[\Delta(t, i)] \, dt$$

where $\Delta(t, i) = v(S_t \cup \{i\}) - v(S_t)$ represents the marginal contribution of data point $i$ when added to a random coalition $S_t$ of size determined by $t \cdot (N-1)$ using configurable rounding methods.

## Key Advantages

### 1. Computational Efficiency
- **Reduced sampling complexity**: Instead of sampling across all $N$ coalition sizes, we sample only a small number of $t$ values
- **Smoothness exploitation**: Leverages the smoothness of $\mathbb{E}[\Delta(t, i)]$ for efficient approximation
- **Scalable to large datasets**: Monte Carlo integration scales much better than exhaustive enumeration

### 2. Flexible Sampling Strategies
- **Uniform sampling**: Simple random sampling of $t$ values
- **Adaptive sampling**: Automatic determination of sampling density based on function smoothness
- **Importance sampling**: Focus sampling on regions where the integrand changes most rapidly

### 3. Multiple Integration Methods
- **Trapezoidal rule**: Simple and robust numerical integration
- **Gaussian quadrature**: High-precision integration with adaptive node selection
- **Monte Carlo integration**: Stochastic approximation suitable for high-dimensional problems

## Features

### Core Functionality
- **Integral Shapley Values**: Efficient computation using continuous integration
- **Multiple Computation Methods**:
  - Monte Carlo sampling
  - Trapezoidal integration
  - Gaussian-Legendre quadrature
  - Adaptive integration with smoothness detection

### Utility Functions
- **RKHS-based similarity**: Reproducing Kernel Hilbert Space similarity measures
- **KL divergence**: Probability distribution similarity  
- **Accuracy-based**: Direct accuracy comparison
- **Cosine similarity**: Parameter vector similarity

### Supported Models
- Support Vector Machines (Linear and RBF kernels)
- Logistic Regression
- Various scikit-learn compatible models

### Datasets
- Iris, Wine, Breast Cancer datasets
- Synthetic datasets for controlled experiments
- Custom data loading support

## Project Structure

```
integral_shapley/
├── src/
│   ├── core/
│   │   ├── integral_shapley.py    # Main computation engine
│   │   └── integration_methods.py # Various integration techniques
│   ├── utils/
│   │   ├── utilities.py          # Utility functions (RKHS, KL, accuracy)
│   │   └── model_utils.py        # Model factory and helpers
│   └── experiments/
│       ├── efficiency_study.py   # Performance comparison experiments
│       ├── smoothness_analysis.py # Study integrand smoothness properties
│       └── convergence_study.py  # Convergence analysis of different methods
├── data/                         # Datasets
├── results/                      # Experimental results
│   ├── pickles/                 # Saved computation results (.pkl)
│   ├── csvs/                    # CSV exports
│   └── plots/                   # Visualization results
├── tests/                       # Unit tests
└── docs/                        # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd integral_shapley
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

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

# Train final model
final_model = return_model('LinearSVC')
final_model.fit(x_train, y_train)

# Create classifier for subsets
clf = return_model('LinearSVC')

# Compute Shapley value for data point 0 using integral method
shapley_value = compute_integral_shapley_value(
    x_train, y_train, x_valid, y_valid, 
    i=0, clf=clf, final_model=final_model,
    utility_func=utility_acc,
    num_t_samples=50, num_MC=100,
    method='trapezoid'
)
print(f"Integral Shapley value for data point 0: {shapley_value}")
```

### Command Line Usage

```bash
# Compute Shapley values using trapezoidal integration
python src/core/integral_shapley.py --dataset iris --utility acc --method trapezoid --num_t_samples 50 --num_MC 100

# Use adaptive sampling based on smoothness detection
python src/core/integral_shapley.py --dataset wine --utility rkhs --method adaptive --tolerance 1e-4

# High-precision Gaussian quadrature
python src/core/integral_shapley.py --dataset cancer --utility kl --method gaussian --num_nodes 32

# Use probabilistic rounding for theoretical accuracy
python src/core/integral_shapley.py --dataset iris --utility acc --method trapezoid --rounding_method probabilistic
```

### Available Options

- **Datasets**: `iris`, `wine`, `cancer`, `synthetic`
- **Utilities**: `rkhs`, `kl`, `acc`, `cosine`
- **Methods**: `trapezoid`, `gaussian`, `adaptive`, `monte_carlo`, `exact`
- **Classifiers**: `svm`, `lr`
- **Rounding Methods**: `probabilistic` (unbiased), `round` (standard), `floor`, `ceil`

## Research Applications

### 1. Computational Efficiency Analysis
Compare the computational cost of integral methods vs. traditional enumeration-based approaches.

### 2. Smoothness Studies
Analyze the smoothness properties of $\mathbb{E}[\Delta(t, i)]$ across different datasets and models.

### 3. Convergence Analysis
Study the convergence rates of different integration methods as a function of sampling density.

### 4. Data Valuation
Assess the contribution of individual training samples to model performance with unprecedented efficiency.

## Mathematical Foundation

The integral formulation transforms the classical Shapley value:

$$\mathcal{SV}_i = \frac{1}{N} \sum_{s=0}^{N-1} \mathbb{E}[\Delta_s]$$

into a continuous integral that can be efficiently approximated using numerical integration techniques, taking advantage of the smoothness properties of the integrand.

### Coalition Size Rounding

A critical implementation detail is how to convert continuous coalition proportions $t \in [0,1]$ to discrete coalition sizes $s \in \{0,1,\ldots,N-1\}$. We provide four methods:

1. **Probabilistic Rounding** (Default): For $x = t \cdot (N-1)$, choose $\lfloor x \rfloor$ with probability $\lceil x \rceil - x$ and $\lceil x \rceil$ with probability $x - \lfloor x \rfloor$. This method is theoretically unbiased: $\mathbb{E}[\text{rounded}(x)] = x$.

2. **Standard Rounding**: $s = \text{round}(t \cdot (N-1))$. Simple and practical with minimal bias.

3. **Floor/Ceil Rounding**: Always round down or up. These introduce systematic bias toward smaller or larger coalitions respectively.

**Recommendation**: Use `probabilistic` rounding for research applications requiring theoretical rigor, and `round` for practical applications.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add appropriate license information]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add appropriate citation information]
```

## References

- Shapley, Lloyd S. "A value for n-person games." (1953)
- Ghorbani, Amirata, and James Zou. "Data shapley: Equitable valuation of machine learning data." (2019)
- [Add other relevant references]