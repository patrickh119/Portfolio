import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Dict, List


class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # annualized
        self.cov_matrix = returns.cov() * 252  # annualized
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns.columns)
        self.assets = returns.columns.tolist()

    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        portfolio_return = np.sum(self.mean_returns * weights)  # weighted sum of individual returns on each asset
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))  # weights variance of portfolio with covariance of assets
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe_ratio

    def negative_sharpe(self, weights: np.ndarray) -> float:
        return -self.portfolio_stats(weights)[2]

    def max_sharpe_portfolio(self) -> Dict:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # weights sum to 1
        bounds = tuple((0, 1) for _ in range(self.num_assets))  # weights between 0 and 1
        initial_weights = self.num_assets * [1. / self.num_assets]

        result = minimize(self.negative_sharpe, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_stats(optimal_weights)

        return {
            'Weights': dict(zip(self.assets, optimal_weights)),
            'Return': ret,
            'Volatility': vol,
            'Sharpe Ratio': sharpe
        }

    def min_volatility_portfolio(self) -> Dict:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = self.num_assets * [1. / self.num_assets]

        results = minimize(self.portfolio_volatility, initial_weights, method='SLSQP',
                           bounds=bounds, constraints=constraints)

        optimal_weights = results.x
        ret, vol, sharpe = self.portfolio_stats(optimal_weights)
        return {
            'Weights': dict(zip(self.assets, optimal_weights)),
            'Return': ret,
            'Volatility': vol,
            'Sharpe Ratio': sharpe
        }

    def efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        min_vol_port = self.min_volatility_portfolio()
        max_sharpe_port = self.max_sharpe_portfolio()

        min_return = min_vol_port['Return']
        max_return = max_sharpe_port['Return']

        target_returns = np.linspace(min_return, max_return, num_portfolios)

        efficient_portfolios = []

        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_stats(x)[0] - target}
            ]

            bounds = tuple((0, 1) for _ in range(self.num_assets))
            initial_weights = self.num_assets * [1. / self.num_assets]

            result = minimize(self.portfolio_volatility, initial_weights, method='SLSQP',
                              bounds=bounds, constraints=constraints, options={'maxiter': 1000}
                              )
            if result.success:
                weights = result.x
                ret, vol, sharpe = self.portfolio_stats(weights)

                portfolio_data = {
                    'Return': ret,
                    'Volatility': vol,
                    'Sharpe Ratio': sharpe
                }
                for i, asset in enumerate(self.assets):
                    portfolio_data[f'weight_{asset}'] = weights[i]

                efficient_portfolios.append(portfolio_data)
        return pd.DataFrame(efficient_portfolios)

    def random_portfolios(self, num_portfolios: int = 10000) -> pd.DataFrame:
        results = []
        for _ in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)

            ret, vol, sharpe = self.portfolio_stats(weights)

            results.append({
                'Return': ret,
                'Volatility': vol,
                'Sharpe Ratio': sharpe
            })

        return pd.DataFrame(results)

    def target_return_portfolio(self, target_return: float) -> Dict:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return}
        ]

        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = np.array([1 / self.num_assets] * self.num_assets)

        result = minimize(self.portfolio_volatility, initial_weights, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization did not converge for target return {target_return}")
        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_stats(optimal_weights)
        return {
            'Weights': dict(zip(self.assets, optimal_weights)),
            'Return': ret,
            'Volatility': vol,
            'Sharpe Ratio': sharpe
        }

    def get_portfolio_summary(self) -> str:
        max_sharpe = self.max_sharpe_portfolio()
        min_vol = self.min_volatility_portfolio()

        summary = "-" * 60 + "\n"
        summary += "PORTFOLIO OPTIMIZATION SUMMARY\n"
        summary += "-" * 60 + "\n\n"

        summary += "MAX SHARPE RATIO PORTFOLIO (Best Risk-Adjusted Returns)\n"
        summary += "-" * 60 + "\n"
        summary += f"Expected Return: {max_sharpe['Return'] * 100:.2f}%\n"
        summary += f"Volatility (Risk): {max_sharpe['Volatility'] * 100:.2f}%\n"
        summary += f"Sharpe Ratio: {max_sharpe['Sharpe Ratio']:.3f}\n"
        summary += "\nAllocation:\n"
        for asset, weight in max_sharpe['Weights'].items():
            summary += f"  {asset}: {weight * 100:.2f}%\n"

        summary += "\n" + "=" * 60 + "\n"
        summary += "MINIMUM VOLATILITY PORTFOLIO (Lowest Risk)\n"
        summary += "-" * 60 + "\n"
        summary += f"Expected Return: {min_vol['Return'] * 100:.2f}%\n"
        summary += f"Volatility (Risk): {min_vol['Volatility'] * 100:.2f}%\n"
        summary += f"Sharpe Ratio: {min_vol['Sharpe Ratio']:.3f}\n"
        summary += "\nAllocation:\n"
        for asset, weight in min_vol['Weights'].items():
            summary += f"  {asset}: {weight * 100:.2f}%\n"

        return summary

    if __name__ == "__main__":
        print("Portfolio Optimizer Loaded - Example Usage\n")

        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=252)

        assets = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D']

        returns_data = {}

        for asset in assets:
            mean_return = np.random.uniform(0.0001, 0.0008)
            volatility = np.random.uniform(0.01, 0.03)
            returns_data[asset] = np.random.normal(mean_return, volatility, len(dates))

        returns_df = pd.DataFrame(returns_data, index=dates)

        optimizer = PortfolioOptimizer(returns_df, risk_free_rate=0.02)

        print(optimizer.get_portfolio_summary())

        frontier = optimizer.efficient_frontier(num_portfolios=50)
        print("\nEfficient Frontier Sample:\n",
              frontier[['Return', 'Volatility', 'Sharpe Ratio']].head())