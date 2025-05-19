import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

# Set plotting style
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

class TradingStrategy:
    def __init__(self, ticker="AMZN", period="6mo", interval="1d"):
        """
        Initialize the trading strategy with default parameters
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (default: AMZN for Amazon)
        period : str
            Time period for data (default: 6mo for 6 months)
        interval : str
            Data interval (default: 1d for daily)
        """
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.data = None
        self.signals = None
        self.performance = {}
    
    def fetch_data(self):
        """Fetch stock data using yfinance"""
        print(f"Fetching {self.ticker} data for the past {self.period}...")
        self.data = yf.download(self.ticker, period=self.period, interval=self.interval)
        print(f"Downloaded {len(self.data)} rows of data")
        return self.data
    
    def calculate_indicators(self, fast_ema=12, slow_ema=26, adx_period=14, rsi_period=14):
        """
        Calculate technical indicators: EMA, ADX, and RSI
        
        Parameters:
        -----------
        fast_ema : int
            Fast EMA period (default: 12)
        slow_ema : int
            Slow EMA period (default: 26)
        adx_period : int
            ADX period (default: 14)
        rsi_period : int
            RSI period (default: 14)
        """
        if self.data is None:
            self.fetch_data()
        
        # Calculate EMAs
        self.data['EMA_fast'] = self.data['Close'].ewm(span=fast_ema, adjust=False).mean()
        self.data['EMA_slow'] = self.data['Close'].ewm(span=slow_ema, adjust=False).mean()
        
        # Calculate RSI
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate ADX
        # True Range
        self.data['TR'] = np.maximum(
            self.data['High'] - self.data['Low'],
            np.maximum(
                abs(self.data['High'] - self.data['Close'].shift(1)),
                abs(self.data['Low'] - self.data['Close'].shift(1))
            )
        )
        
        # Directional Movement
        self.data['DM+'] = np.where(
            (self.data['High'] - self.data['High'].shift(1)) > (self.data['Low'].shift(1) - self.data['Low']),
            np.maximum(self.data['High'] - self.data['High'].shift(1), 0),
            0
        )
        
        self.data['DM-'] = np.where(
            (self.data['Low'].shift(1) - self.data['Low']) > (self.data['High'] - self.data['High'].shift(1)),
            np.maximum(self.data['Low'].shift(1) - self.data['Low'], 0),
            0
        )
        
        # Smoothed TR and DM
        self.data['ATR'] = self.data['TR'].rolling(window=adx_period).mean()
        self.data['DI+'] = 100 * (self.data['DM+'].rolling(window=adx_period).mean() / self.data['ATR'])
        self.data['DI-'] = 100 * (self.data['DM-'].rolling(window=adx_period).mean() / self.data['ATR'])
        
        # Directional Index
        self.data['DX'] = 100 * (abs(self.data['DI+'] - self.data['DI-']) / (self.data['DI+'] + self.data['DI-']))
        
        # Average Directional Index
        self.data['ADX'] = self.data['DX'].rolling(window=adx_period).mean()
        
        # Drop NaN values
        self.data.dropna(inplace=True)
        
        return self.data
    
    def generate_signals(self, adx_threshold=25, rsi_buy=30, rsi_sell=70):
        """
        Generate buy/sell signals based on EMA crossover, ADX, and RSI
        
        Parameters:
        -----------
        adx_threshold : int
            ADX threshold value (default: 25)
        rsi_buy : int
            RSI oversold threshold for buy signal (default: 30)
        rsi_sell : int
            RSI overbought threshold for sell signal (default: 70)
        """
        if 'ADX' not in self.data.columns:
            self.calculate_indicators()
        
        # Initialize signal column
        self.data['Signal'] = 0
        
        # Buy signal: Fast EMA crosses above Slow EMA, ADX > threshold, RSI < rsi_buy
        buy_condition = (
            (self.data['EMA_fast'] > self.data['EMA_slow']) & 
            (self.data['EMA_fast'].shift(1) <= self.data['EMA_slow'].shift(1)) &
            (self.data['ADX'] > adx_threshold) &
            (self.data['RSI'] < rsi_buy)
        )
        self.data.loc[buy_condition, 'Signal'] = 1
        
        # Sell signal: Fast EMA crosses below Slow EMA, ADX > threshold, RSI > rsi_sell
        sell_condition = (
            (self.data['EMA_fast'] < self.data['EMA_slow']) & 
            (self.data['EMA_fast'].shift(1) >= self.data['EMA_slow'].shift(1)) &
            (self.data['ADX'] > adx_threshold) &
            (self.data['RSI'] > rsi_sell)
        )
        self.data.loc[sell_condition, 'Signal'] = -1
        
        # Create a copy for the signals dataframe
        self.signals = self.data[self.data['Signal'] != 0].copy()
        
        return self.signals
    
    def backtest_strategy(self, initial_capital=10000):
        """
        Backtest the trading strategy
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for the backtest (default: 10000)
        """
        if self.signals is None:
            self.generate_signals()
        
        # Copy signals to a separate variable to avoid modification
        signals = self.data['Signal'].values
        
        # Initialize position array with zeros
        positions = np.zeros(len(signals))
        
        # Simple logic: Start with no position, buy on signal 1, sell on signal -1
        position = 0  # Start with no position
        
        # Loop through each day and update position
        for i in range(len(signals)):
            # Update position based on signals
            if signals[i] == 1:  # Buy signal
                position = 1
            elif signals[i] == -1:  # Sell signal
                position = 0
            
            # Store the position for this day
            positions[i] = position
        
        # Add positions to dataframe
        self.data['Position'] = positions
        
        # Calculate price changes
        self.data['Price_Change'] = self.data['Close'].pct_change().fillna(0)
        
        # Calculate strategy returns (today's return = yesterday's position * today's price change)
        self.data['Strategy_Return'] = self.data['Position'].shift(1) * self.data['Price_Change']
        self.data['Strategy_Return'].fillna(0, inplace=True)  # Fill NaN for first day
        
        # Calculate market returns (buy and hold)
        self.data['Market_Return'] = self.data['Price_Change']
        
        # Calculate cumulative returns
        self.data['Cumulative_Market_Return'] = (1 + self.data['Market_Return']).cumprod()
        self.data['Cumulative_Strategy_Return'] = (1 + self.data['Strategy_Return']).cumprod()
        
        # Calculate portfolio value
        self.data['Portfolio_Value'] = initial_capital * self.data['Cumulative_Strategy_Return']
        
        # Calculate performance metrics
        total_return = self.data['Cumulative_Strategy_Return'].iloc[-1] - 1
        market_return = self.data['Cumulative_Market_Return'].iloc[-1] - 1
        
        # Annualize returns (approximate)
        days = (self.data.index[-1] - self.data.index[0]).days
        if days > 0:
            annual_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annual_return = 0
        
        # Calculate Sharpe Ratio (assuming risk free rate = 0 for simplicity)
        if self.data['Strategy_Return'].std() > 0:
            sharpe_ratio = annual_return / (self.data['Strategy_Return'].std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        rolling_max = self.data['Portfolio_Value'].cummax()
        drawdown = (self.data['Portfolio_Value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Store performance metrics
        self.performance = {
            'Total Return': f"{total_return:.2%}",
            'Market Return': f"{market_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Number of Trades': len(self.signals)
        }
        
        return self.performance
    
    def plot_results(self):
        """Plot the strategy results"""
        if 'Portfolio_Value' not in self.data.columns:
            self.backtest_strategy()
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
        
        # Plot 1: Price with EMAs and signals
        axs[0].plot(self.data.index, self.data['Close'], label='Close Price', alpha=0.7)
        axs[0].plot(self.data.index, self.data['EMA_fast'], label=f'Fast EMA', color='orange')
        axs[0].plot(self.data.index, self.data['EMA_slow'], label=f'Slow EMA', color='red')
        
        # Buy signals
        buy_signals = self.data[self.data['Signal'] == 1]
        axs[0].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal')
        
        # Sell signals
        sell_signals = self.data[self.data['Signal'] == -1]
        axs[0].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal')
        
        axs[0].set_title(f'{self.ticker} Price with Trading Signals')
        axs[0].set_ylabel('Price ($)')
        axs[0].legend()
        
        # Plot 2: ADX and RSI indicators
        axs[1].plot(self.data.index, self.data['ADX'], label='ADX', color='purple')
        axs[1].axhline(y=25, color='gray', linestyle='--', alpha=0.5)
        axs[1].set_ylabel('ADX')
        axs[1].legend(loc='upper left')
        
        ax1_twin = axs[1].twinx()
        ax1_twin.plot(self.data.index, self.data['RSI'], label='RSI', color='blue')
        ax1_twin.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax1_twin.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax1_twin.set_ylabel('RSI')
        ax1_twin.legend(loc='upper right')
        
        # Plot 3: Strategy performance
        axs[2].plot(self.data.index, self.data['Cumulative_Market_Return'], label='Buy and Hold', color='blue')
        axs[2].plot(self.data.index, self.data['Cumulative_Strategy_Return'], label='Strategy', color='green')
        axs[2].set_title('Strategy Performance')
        axs[2].set_ylabel('Cumulative Return')
        axs[2].legend()
        axs[2].set_xlabel('Date')
        
        # Add performance metrics as text
        perf_text = "\n".join([f"{k}: {v}" for k, v in self.performance.items()])
        plt.figtext(0.15, 0.01, perf_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        return fig
    
    def summary(self):
        """Print a summary of the strategy and performance"""
        if not self.performance:
            self.backtest_strategy()
        
        print(f"\n{'='*50}")
        print(f"TRADING STRATEGY SUMMARY FOR {self.ticker}")
        print(f"{'='*50}")
        print(f"Period: {self.period}, Interval: {self.interval}")
        print(f"Data Range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"\nPerformance Metrics:")
        for key, value in self.performance.items():
            print(f"- {key}: {value}")
        
        print(f"\nTrading Signals:")
        print(f"- Buy signals: {len(self.signals[self.signals['Signal'] == 1])}")
        print(f"- Sell signals: {len(self.signals[self.signals['Signal'] == -1])}")
        print(f"{'='*50}\n")


# Example usage
if __name__ == "__main__":
    # Create strategy instance
    amzn_strategy = TradingStrategy(ticker="AMZN", period="6mo", interval="1d")
    
    # Fetch data
    amzn_strategy.fetch_data()
    
    # Calculate indicators
    amzn_strategy.calculate_indicators(fast_ema=12, slow_ema=26, adx_period=14, rsi_period=14)
    
    # Generate signals
    amzn_strategy.generate_signals(adx_threshold=25, rsi_buy=30, rsi_sell=70)
    
    # Backtest strategy
    amzn_strategy.backtest_strategy(initial_capital=10000)
    
    # Print summary
    amzn_strategy.summary()
    
    # Plot results
    fig = amzn_strategy.plot_results()
    plt.show()