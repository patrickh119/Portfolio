# Portfolio Optimizer

## Overview
The Portfolio Optimizer is a Python-based tool designed to assist in the optimization of investment portfolios. It provides various methods for calculating portfolio statistics, maximizing the Sharpe ratio, minimizing volatility, generating the efficient frontier, and creating random portfolios.

## Features
- **Portfolio Statistics**: Calculate expected returns, volatility, and Sharpe ratio for a given set of asset weights.
- **Maximize Sharpe Ratio**: Identify the optimal asset allocation that provides the best risk-adjusted returns.
- **Minimize Volatility**: Find the asset allocation that minimizes portfolio risk.
- **Efficient Frontier**: Generate a set of optimal portfolios that offer the highest expected return for a given level of risk.
- **Random Portfolios**: Create random portfolios for visualization and analysis.

## Installation
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

## Usage
1. Prepare your asset return data in a pandas DataFrame format.
2. Instantiate the `PortfolioOptimizer` class with the returns DataFrame and an optional risk-free rate.
3. Use the provided methods to optimize your portfolio.

### Example
```python
import numpy as np
import pandas as pd
from portfolio_optimizer import PortfolioOptimizer

# Sample data generation
dates = pd.date_range(start='2020-01-01', periods=252)
returns_data = {
    'Asset_A': np.random.normal(0.0005, 0.02, len(dates)),
    'Asset_B': np.random.normal(0.0004, 0.015, len(dates)),
    'Asset_C': np.random.normal(0.0006, 0.025, len(dates)),
    'Asset_D': np.random.normal(0.0003, 0.01, len(dates)),
}
returns_df = pd.DataFrame(returns_data, index=dates)

# Portfolio optimization
optimizer = PortfolioOptimizer(returns_df)
summary = optimizer.get_portfolio_summary()
print(summary)
```

## Testing
Unit tests for the `PortfolioOptimizer` class can be found in the `tests/test_portfolio_optimizer.py` file. To run the tests, use the following command:

```
pytest tests/
```
