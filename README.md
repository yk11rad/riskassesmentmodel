# riskassesmentmodel
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
