ExportPublishTechnical Trading Strategy
A Python-based trading strategy implementation that combines EMA crossovers with ADX and RSI indicators for market analysis and automated signal generation.
Overview
This project implements a comprehensive trading strategy that utilizes technical analysis indicators to identify potential entry and exit points in financial markets. The strategy combines three primary indicators:

Exponential Moving Average (EMA) Crossovers: To identify trend direction changes
Average Directional Index (ADX): To confirm trend strength
Relative Strength Index (RSI): To identify overbought/oversold conditions

The implementation includes functionality for data acquisition, indicator calculation, signal generation, backtesting, and visualization of results.
Features

📈 Data Acquisition: Fetches financial data from Yahoo Finance API
📊 Technical Analysis: Calculates EMA, ADX, and RSI indicators
🔍 Signal Generation: Identifies buy/sell signals based on indicator combinations
📉 Backtesting: Evaluates strategy performance with key metrics
📋 Performance Reporting: Provides detailed performance statistics
📊 Visualization: Creates multi-panel charts for strategy analysis

Requirements
pandas
numpy
matplotlib
yfinance
seaborn


Installation

Clone this repository and install the required dependencies:
bashgit clone https://github.com/yourusername/technical-trading-strategy.git
cd technical-trading-strategy
pip install -r requirements.txt
Usage
pythonfrom trading_strategy import TradingStrategy

# Create a new strategy instance
strategy = TradingStrategy(ticker="AAPL", period="1y", interval="1d")

# Run the full analysis
strategy.fetch_data()
strategy.calculate_indicators()
strategy.generate_signals()
strategy.backtest_strategy()

# Display results
strategy.summary()
strategy.plot_results()
Configuration
The strategy can be customized with various parameters:
Initialization

ticker: Stock symbol (default: "AMZN")
period: Time period for analysis (default: "6mo")
interval: Data interval (default: "1d")

Indicator Parameters

fast_ema: Fast EMA period (default: 12)
slow_ema: Slow EMA period (default: 26)
adx_period: ADX calculation period (default: 14)
rsi_period: RSI calculation period (default: 14)

Signal Generation

adx_threshold: Minimum ADX value for trend confirmation (default: 25)
rsi_buy: RSI level for buy signals (default: 30)
rsi_sell: RSI level for sell signals (default: 70)

Backtesting

initial_capital: Starting capital for backtesting (default: 10000)

Performance Metrics
The strategy calculates and reports the following performance metrics:

Total Return: Overall percentage gain/loss
Market Return: Buy-and-hold comparison
Annual Return: Annualized performance
Sharpe Ratio: Risk-adjusted return
Max Drawdown: Largest peak-to-trough decline
Number of Trades: Total trading activity

Example Output
==================================================
TRADING STRATEGY SUMMARY FOR AMZN
==================================================
Period: 6mo, Interval: 1d
Data Range: 2024-11-19 to 2025-05-19

Performance Metrics:
- Total Return: 12.45%
- Market Return: 8.92%
- Annual Return: 26.38%
- Sharpe Ratio: 1.84
- Max Drawdown: -5.67%
- Number of Trades: 8

Trading Signals:
- Buy signals: 5
- Sell signals: 3
==================================================
Visualization
The plot_results() method generates a three-panel visualization:

Price Chart: Displays price action, EMAs, and buy/sell signals
Technical Indicators: Shows ADX and RSI with threshold lines
Performance Comparison: Compares strategy returns vs. buy-and-hold

License
This project is licensed under the MIT License - see the LICENSE file for details.
Disclaimer
This software is for educational and informational purposes only. It is not intended as financial advice. Trading involves risk, and past performance is not indicative of future results.
