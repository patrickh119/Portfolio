import numpy as np
import pandas as pd
import pytest
from portfolio_optimizer import PortfolioOptimizer

@pytest.fixture
def sample_returns():
    dates = pd.date_range(start='2020-01-01', periods=252)
    returns_data = {
        'Asset_A': np.random.normal(0.0005, 0.02, len(dates)),
        'Asset_B': np.random.normal(0.0006, 0.02, len(dates)),
        'Asset_C': np.random.normal(0.0004, 0.02, len(dates)),
        'Asset_D': np.random.normal(0.0007, 0.02, len(dates)),
    }
    return pd.DataFrame(returns_data, index=dates)

def test_portfolio_stats(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    ret, vol, sharpe = optimizer.portfolio_stats(weights)
    
    assert isinstance(ret, float)
    assert isinstance(vol, float)
    assert isinstance(sharpe, float)

def test_negative_sharpe(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    neg_sharpe = optimizer.negative_sharpe(weights)
    
    assert isinstance(neg_sharpe, float)

def test_max_sharpe_portfolio(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    result = optimizer.max_sharpe_portfolio()
    
    assert 'Weights' in result
    assert 'Return' in result
    assert 'Volatility' in result
    assert 'Sharpe Ratio' in result

def test_min_volatility_portfolio(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    result = optimizer.min_volatility_portfolio()
    
    assert 'Weights' in result
    assert 'Return' in result
    assert 'Volatility' in result
    assert 'Sharpe Ratio' in result

def test_efficient_frontier(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    frontier = optimizer.efficient_frontier(num_portfolios=50)
    
    assert isinstance(frontier, pd.DataFrame)
    assert frontier.shape[0] == 50

def test_random_portfolios(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    random_results = optimizer.random_portfolios(num_portfolios=10000)
    
    assert isinstance(random_results, pd.DataFrame)
    assert random_results.shape[0] == 10000

def test_target_return_portfolio(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    target_return = 0.0005
    result = optimizer.target_return_portfolio(target_return)
    
    assert 'Weights' in result
    assert 'Return' in result
    assert 'Volatility' in result
    assert 'Sharpe Ratio' in result

def test_get_portfolio_summary(sample_returns):
    optimizer = PortfolioOptimizer(sample_returns)
    summary = optimizer.get_portfolio_summary()
    
    assert isinstance(summary, str)