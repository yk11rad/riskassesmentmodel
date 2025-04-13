#!/usr/bin/env python3
"""
Portfolio Risk Assessment Model
==============================
A sophisticated Monte Carlo simulation for evaluating financial risk in investment portfolios, using GBP (£) as the currency.
Developed for professional submission to demonstrate expertise in quantitative finance and Python programming.
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_t
from scipy.optimize import minimize
from scipy.stats import kstest

class PortfolioRiskModel:
    """A class for Monte Carlo-based portfolio risk assessment with advanced features."""
    
    @staticmethod
    def simulate_portfolio(
        weights, mean_returns, cov_matrix, initial_investment, num_simulations=10000,
        time_horizon=1, distribution='normal', df=3
    ):
        """
        Simulate portfolio outcomes using Monte Carlo with flexible distributions.
        
        Args:
            weights (np.ndarray): Portfolio weights (sum to 1).
            mean_returns (np.ndarray): Expected annual returns per asset.
            cov_matrix (np.ndarray): Covariance matrix of asset returns.
            initial_investment (float): Initial portfolio value in GBP.
            num_simulations (int): Number of simulations.
            time_horizon (float): Investment horizon in years.
            distribution (str): 'normal' or 't' (Student's t for fat tails).
            df (float): Degrees of freedom for t-distribution.
        
        Returns:
            tuple: Simulated portfolio values and returns.
        """
        if distribution == 'normal':
            sim_returns = np.random.multivariate_normal(
                mean_returns * time_horizon, cov_matrix * time_horizon, num_simulations
            )
        elif distribution == 't':
            sim_returns = multivariate_t.rvs(
                loc=mean_returns * time_horizon, shape=cov_matrix * time_horizon,
                df=df, size=num_simulations
            )
        else:
            raise ValueError("Distribution must be 'normal' or 't'")
        
        portfolio_returns = np.dot(sim_returns, weights)
        portfolio_values = initial_investment * (1 + portfolio_returns)
        return portfolio_values, portfolio_returns
    
    @staticmethod
    def apply_stress_scenario(portfolio_values, portfolio_returns, initial_investment, scenario='2008 Crisis'):
        """
        Apply stress scenarios to simulated portfolio outcomes.
        
        Args:
            portfolio_values (np.ndarray): Simulated portfolio values in GBP.
            portfolio_returns (np.ndarray): Simulated portfolio returns.
            initial_investment (float): Initial portfolio value in GBP.
            scenario (str): Stress scenario ('2008 Crisis' or 'COVID Crash').
        
        Returns:
            np.ndarray: Stressed portfolio values in GBP.
        """
        stress_shocks = {'2008 Crisis': -0.5, 'COVID Crash': -0.3}
        shock = stress_shocks.get(scenario, 0)
        stressed_returns = portfolio_returns * (1 + shock)
        return initial_investment * (1 + stressed_returns)
    
    @staticmethod
    def calculate_risk_metrics(portfolio_values, portfolio_returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        
        Args:
            portfolio_values (np.ndarray): Simulated portfolio values in GBP.
            portfolio_returns (np.ndarray): Simulated portfolio returns.
            confidence_level (float): Confidence level for VaR and CVaR.
        
        Returns:
            tuple: VaR and CVaR values in GBP.
        """
        var_percentile = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        var = -var_percentile * initial_investment
        cvar = -np.mean(portfolio_returns[portfolio_returns <= var_percentile]) * initial_investment
        return var, cvar
    
    @staticmethod
    def compute_risk_contribution(weights, cov_matrix, initial_investment):
        """
        Compute each asset's marginal contribution to portfolio risk.
        
        Args:
            weights (np.ndarray): Portfolio weights.
            cov_matrix (np.ndarray): Covariance matrix.
            initial_investment (float): Initial portfolio value in GBP.
        
        Returns:
            np.ndarray: Risk contributions per asset in GBP.
        """
        portfolio_var = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_contrib = (cov_matrix @ weights) / portfolio_var
        return marginal_contrib * initial_investment
    
    @staticmethod
    def optimize_portfolio(mean_returns, cov_matrix, risk_aversion=0.5):
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            mean_returns (np.ndarray): Expected returns.
            cov_matrix (np.ndarray): Covariance matrix.
            risk_aversion (float): Risk aversion parameter.
        
        Returns:
            np.ndarray: Optimized weights.
        """
        n = len(mean_returns)
        initial_weights = np.ones(n) / n
        
        def objective(weights):
            ret = np.sum(weights * mean_returns)
            risk = np.sqrt(weights.T @ cov_matrix @ weights)
            return -ret + risk_aversion * risk**2
        
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w}
        )
        bounds = [(0, 1) for _ in range(n)]
        
        result = minimize(
            objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints
        )
        return result.x
    
    @staticmethod
    def validate_distribution(returns, confidence=0.95):
        """
        Validate return distribution using Kolmogorov-Smirnov test.
        
        Args:
            returns (np.ndarray): Portfolio returns.
            confidence (float): Confidence level for test.
        
        Returns:
            str: Validation result message.
        """
        stat, p = kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
        return (f"Distribution normality rejected (p={p:.4f})" if p < 1 - confidence
                else f"Normality not rejected (p={p:.4f})")
    
    @staticmethod
    def visualize_results(portfolio_values, portfolio_returns, stressed_values, var, cvar, initial_investment, weights, cov_matrix, mean_returns):
        """
        Visualize portfolio outcomes and risk metrics in GBP.
        
        Args:
            portfolio_values (np.ndarray): Simulated portfolio values in GBP.
            portfolio_returns (np.ndarray): Simulated portfolio returns.
            stressed_values (np.ndarray): Stressed portfolio values in GBP.
            var (float): Value at Risk in GBP.
            cvar (float): Conditional Value at Risk in GBP.
            initial_investment (float): Initial portfolio value in GBP.
            weights (np.ndarray): Portfolio weights.
            cov_matrix (np.ndarray): Covariance matrix.
            mean_returns (np.ndarray): Expected returns.
        """
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 2, 1)
        sns.histplot(portfolio_values, bins=50, kde=True, color='blue', label='Base Scenario')
        sns.histplot(stressed_values, bins=50, kde=True, color='red', alpha=0.5, label='Stressed Scenario')
        plt.axvline(initial_investment - var, color='black', linestyle='--', label=f'VaR: £{var:,.2f}')
        plt.axvline(initial_investment - cvar, color='orange', linestyle='--', label=f'CVaR: £{cvar:,.2f}')
        plt.title('Portfolio Value Distribution')
        plt.xlabel('Portfolio Value (£)')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        sns.histplot(portfolio_returns * 100, bins=50, kde=True, color='green')
        plt.axvline(-var / initial_investment * 100, color='black', linestyle='--', label=f'VaR: {var/initial_investment*100:.2f}%')
        plt.axvline(-cvar / initial_investment * 100, color='orange', linestyle='--', label=f'CVaR: {cvar/initial_investment*100:.2f}%')
        plt.title('Portfolio Return Distribution')
        plt.xlabel('Return (%)')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        contrib = PortfolioRiskModel.compute_risk_contribution(weights, cov_matrix, initial_investment)
        plt.bar(['Asset 1', 'Asset 2', 'Asset 3'], contrib, color='purple')
        plt.title('Marginal Risk Contributions')
        plt.ylabel('Risk Contribution (£)')
        
        plt.subplot(2, 2, 4)
        returns, risks = [], []
        for _ in range(100):
            w = np.random.dirichlet(np.ones(len(weights)))
            returns.append(w @ mean_returns)
            risks.append(np.sqrt(w.T @ cov_matrix @ w))
        plt.scatter(risks, returns, c='blue', alpha=0.5, label='Random Portfolios')
        opt_weights = PortfolioRiskModel.optimize_portfolio(mean_returns, cov_matrix)
        opt_return = opt_weights @ mean_returns
        opt_risk = np.sqrt(opt_weights.T @ cov_matrix @ opt_weights)
        plt.scatter([opt_risk], [opt_return], c='red', s=100, label='Optimal Portfolio')
        plt.title('Efficient Frontier')
        plt.xlabel('Risk (Std Dev)')
        plt.ylabel('Expected Return')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Execute portfolio risk assessment and display results in GBP."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define portfolio parameters
    initial_investment = 100_000  # Initial investment: £100,000
    num_simulations = 10_000     # Number of Monte Carlo simulations
    time_horizon = 1             # Investment horizon: 1 year
    mean_returns = np.array([0.08, 0.12, 0.05])  # Expected returns: 8%, 12%, 5%
    volatilities = np.array([0.15, 0.20, 0.10])  # Volatilities: 15%, 20%, 10%
    correlation_matrix = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.2],
        [0.1, 0.2, 1.0]
    ])  # Correlation matrix
    weights = np.array([0.4, 0.4, 0.2])  # Portfolio weights: 40%, 40%, 20%
    
    # Compute covariance matrix
    cov_matrix = np.diag(volatilities) @ correlation_matrix @ np.diag(volatilities)
    
    # Run Monte Carlo simulation with t-distribution
    portfolio_values, portfolio_returns = PortfolioRiskModel.simulate_portfolio(
        weights=weights,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        initial_investment=initial_investment,
        num_simulations=num_simulations,
        time_horizon=time_horizon,
        distribution='t',
        df=3
    )
    
    # Apply stress scenario
    stressed_values = PortfolioRiskModel.apply_stress_scenario(
        portfolio_values, portfolio_returns, initial_investment, scenario='2008 Crisis'
    )
    
    # Calculate risk metrics
    var, cvar = PortfolioRiskModel.calculate_risk_metrics(portfolio_values, portfolio_returns)
    
    # Validate distribution
    validation_result = PortfolioRiskModel.validate_distribution(portfolio_returns)
    
    # Compute risk contributions
    risk_contributions = PortfolioRiskModel.compute_risk_contribution(weights, cov_matrix, initial_investment)
    
    # Optimize portfolio
    optimal_weights = PortfolioRiskModel.optimize_portfolio(mean_returns, cov_matrix)
    
    # Display results
    print("Portfolio Risk Assessment Results")
    print("=" * 35)
    print(f"Initial Investment: £{initial_investment:,.2f}")
    print(f"Expected Portfolio Value: £{np.mean(portfolio_values):,.2f}")
    print(f"Expected Stressed Value (2008 Crisis): £{np.mean(stressed_values):,.2f}")
    print(f"Portfolio Volatility: £{np.std(portfolio_values):,.2f}")
    print(f"95% Value at Risk (VaR): £{var:,.2f}")
    print(f"95% Conditional Value at Risk (CVaR): £{cvar:,.2f}")
    print(f"Distribution Validation: {validation_result}")
    print(f"Risk Contributions per Asset: {np.round(risk_contributions, 2)}")
    print(f"Optimal Portfolio Weights: {np.round(optimal_weights, 3)}")
    
    # Visualize results
    PortfolioRiskModel.visualize_results(
        portfolio_values, portfolio_returns, stressed_values, var, cvar,
        initial_investment, weights, cov_matrix, mean_returns
    )

if __name__ == "__main__":
    main()

"""
README
======

Overview
--------
This Python script implements an advanced Monte Carlo simulation for assessing financial risk in investment portfolios, using British Pounds (GBP, £) as the currency. The model simulates thousands of portfolio outcomes, incorporating both normal and Student's t-distributions to capture tail risks. It includes stress testing for historical crises, calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR), decomposes portfolio risk by asset, optimizes portfolio weights using mean-variance optimization, and validates return distributions with statistical tests. Comprehensive visualizations provide clear, actionable insights for portfolio management.

Methodology
-----------
The model employs a Monte Carlo simulation framework to generate portfolio outcomes based on random asset returns. Key components include:
- **Distribution Modeling**: Supports normal and t-distributions, with the latter capturing fat tails for extreme event analysis.
- **Stress Testing**: Applies predefined shocks (e.g., -50% for 2008 Financial Crisis) to simulate crisis impacts.
- **Risk Metrics**: Computes VaR and CVaR at a 95% confidence level to quantify potential losses.
- **Risk Decomposition**: Calculates marginal risk contributions per asset, aiding in risk attribution.
- **Portfolio Optimization**: Uses mean-variance optimization to derive weights maximizing return for a given risk level.
- **Distribution Validation**: Applies the Kolmogorov-Smirnov test to assess return distribution assumptions.
- **Visualization**: Generates four-panel plots showing portfolio value distributions, returns, risk contributions, and the efficient frontier.

The simulation assumes static parameters but is extensible for dynamic inputs, making it suitable for both static and adaptive risk analysis.

Requirements
------------
- **Python Version**: 3.8 or higher
- **Dependencies**:
  - numpy: Numerical computations and array operations.
  - pandas: Data manipulation and analysis.
  - matplotlib: Plotting framework for visualizations.
  - seaborn: Enhanced statistical visualizations.
  - scipy: Statistical distributions and optimization algorithms.
"""
