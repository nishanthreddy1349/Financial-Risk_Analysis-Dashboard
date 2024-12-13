
#<--------------------------------------------Importing the libraries------------------------------------------------------------>
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import date, timedelta
import time
import scipy.optimize as optimize
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
import requests
import os

#<----------------------------------------------Data Fetching Functions---------------------------------------------------------->
def fetch_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
            data[ticker] = stock_data
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
    return data

#<------------------------------------------Performance Metrics Calculations-------------------------------------------------------------->


def calculate_performance_metrics(returns, market_returns, risk_free_rate=0.0):
    metrics = {}

    market_var = np.var(market_returns)

    for ticker in returns.columns:
        ticker_returns = returns[ticker]
        

        volatility = ticker_returns.std() * np.sqrt(252)
        annual_return = ticker_returns.mean() * 252
        

        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        

        X = sm.add_constant(market_returns)  
        y = ticker_returns
        
        model = sm.OLS(y, X).fit()  
        beta = model.params[1]  
        

        cum_returns = (1 + ticker_returns).cumprod()
        drawdowns = (cum_returns - cum_returns.expanding().max()) / cum_returns.expanding().max()
        max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
        

        metrics[ticker] = {
            'Annual Return': annual_return,
            'Annual Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Beta': beta
        }


    return pd.DataFrame(metrics).T

#<-------------------------------------------------Portfolio Analysis------------------------------------------------------->
class PortfolioAnalysis:
    def __init__(self, returns_data, risk_free_rate=0.0):
        self.returns = returns_data
        self.rf = risk_free_rate
        self.cov_matrix = LedoitWolf().fit(returns_data).covariance_
        
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio return and volatility"""
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std

    def negative_sharpe_ratio(self, weights):
        """Calculate negative Sharpe ratio for optimization"""
        p_ret, p_std = self.calculate_portfolio_metrics(weights)
        return -(p_ret - self.rf) / p_std

    def get_efficient_frontier(self, num_portfolios=1000):
        """Generate efficient frontier points"""
        n_assets = len(self.returns.columns)
        returns_range = np.linspace(
            self.returns.mean().min() * 252,
            self.returns.mean().max() * 252,
            num_portfolios
        )
        efficient_portfolios = []
        
        for ret in returns_range:
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.calculate_portfolio_metrics(x)[0] - ret}
            )
            bounds = tuple((0, 1) for _ in range(n_assets))
            result = optimize.minimize(
                lambda x: self.calculate_portfolio_metrics(x)[1],
                n_assets * [1./n_assets],
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            if result.success:
                efficient_portfolios.append({
                    'Return': ret,
                    'Volatility': self.calculate_portfolio_metrics(result.x)[1],
                    'Weights': result.x
                })
                
        return pd.DataFrame(efficient_portfolios)
#<--------------------------------------------------Optimal portfolio------------------------------------------------------>
class OptimalPortfolio:
    def __init__(self, returns_data, risk_free_rate=0.0):
        self.returns = returns_data
        self.rf = risk_free_rate
        self.mean_returns = returns_data.mean() * 252
        self.cov_matrix = returns_data.cov() * 252
        self.num_assets = len(returns_data.columns)
        
    def portfolio_performance(self, weights):
        returns = np.sum(self.mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe = (returns - self.rf) / std if std > 0 else 0
        return returns, std, sharpe
        
    def max_sharpe_ratio(self):
        """Maximize Sharpe Ratio"""
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = [1/self.num_assets] * self.num_assets
        
        result = optimize.minimize(
            lambda x: -self.portfolio_performance(x)[2],
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x
        
    def min_volatility(self):
        """Minimize Volatility"""
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_weights = [1/self.num_assets] * self.num_assets
        
        result = optimize.minimize(
            lambda x: self.portfolio_performance(x)[1],
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x
#<-------------------------------------------------Calculate the CAPM metrics------------------------------------------------------->

def calculate_capm_metrics(stock_returns, market_returns, risk_free_rate=0.0):
    """Calculate CAPM metrics including beta and expected return"""
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    
    market_return = np.mean(market_returns) * 252
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    
    return {
        'Beta': beta,
        'Expected Return (%)': expected_return * 100,
        'Risk Premium (%)': (expected_return - risk_free_rate) * 100
    }

#<-------------------------------------------Visualization and Analysis Functions------------------------------------------------------------->
def create_stock_graphs(df, auto_refresh=True):
    st.subheader("Stock Price Visualization")
    graph_type = st.radio("Select Graph Type", ["Combined Graph", "Individual Graphs"], horizontal=True)
    
    auto_refresh = st.checkbox("Enable Auto-refresh (30 seconds)", value=False)
    
    # Display key metrics
    st.subheader("Current Stock Metrics")
    metrics_cols = st.columns(len(df.columns))
    for idx, column in enumerate(df.columns):
        with metrics_cols[idx]:
            current_price = df[column].iloc[-1]
            price_change = df[column].iloc[-1] - df[column].iloc[-2]
            percent_change = (price_change / df[column].iloc[-2]) * 100
            
            st.metric(
                label=column,
                value=f"${current_price:.2f}",
                delta=f"{percent_change:.2f}%"
            )

    normalized_df = df.div(df.iloc[0]) * 100
    if graph_type == "Combined Graph":
        fig = go.Figure()
        for column in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=normalized_df[column], name=column, mode='lines'))
        fig.update_layout(title="Normalized Stock Prices (Base=100)", xaxis_title="Date", yaxis_title="Normalized Price", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        cols = st.columns(2)
        for idx, column in enumerate(df.columns):
            with cols[idx % 2]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column, mode='lines'))
                fig.update_layout(title=f"{column} Stock Price", xaxis_title="Date", yaxis_title="Price", height=400)
                st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(15)
        st.rerun()

#<-----------------------------------------------Financial Ratios--------------------------------------------------------->
def calculate_financial_ratios(ticker, period="quarterly"):
    """Calculate financial ratios for a given stock"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get income statement
        income_stmt = stock.income_stmt if period == "yearly" else stock.quarterly_income_stmt
        if income_stmt is None or income_stmt.empty:
            return pd.DataFrame()
            
        # Get balance sheet
        balance = stock.balance_sheet if period == "yearly" else stock.quarterly_balance_sheet
        if balance is None or balance.empty:
            return pd.DataFrame()
            
        # Calculate Net Profit Margin
        net_income = income_stmt.loc['Net Income']
        revenue = income_stmt.loc['Total Revenue']
        net_profit_margin = (net_income / revenue).round(4) * 100
        
        # Calculate Return on Equity
        total_equity = balance.loc['Stockholders Equity']
        roe = (net_income / total_equity).round(4) * 100
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Period': income_stmt.columns,
            'Net Profit Margin (%)': net_profit_margin.values,
            'Return on Equity (%)': roe.values
        })
        
        results.set_index('Period', inplace=True)
        results = results.sort_index(ascending=False)
        
        return results
    except Exception as e:
        st.write(f"Error calculating ratios for {ticker}: {str(e)}")
        return pd.DataFrame()

#<-----------------------------------------------Display optimal Allocations --------------------------------------------------------->

def display_optimal_allocations(returns, current_weights):
    """Display optimal portfolio allocations"""
    st.header("Optimal Portfolio Allocation")
    
    optimizer = OptimalPortfolio(returns)
    max_sharpe_weights = optimizer.max_sharpe_ratio()
    min_vol_weights = optimizer.min_volatility()
    
    current_perf = optimizer.portfolio_performance(np.array(list(current_weights.values())))
    max_sharpe_perf = optimizer.portfolio_performance(max_sharpe_weights)
    min_vol_perf = optimizer.portfolio_performance(min_vol_weights)
    
    portfolio_comparisons = pd.DataFrame({
        'Current Portfolio': list(current_weights.values()),
        'Maximum Sharpe Ratio': max_sharpe_weights,
        'Minimum Volatility': min_vol_weights
    }, index=returns.columns)
    
    metrics_comparison = pd.DataFrame({
        'Current Portfolio': [current_perf[0], current_perf[1], current_perf[2]],
        'Maximum Sharpe Ratio': [max_sharpe_perf[0], max_sharpe_perf[1], max_sharpe_perf[2]],
        'Minimum Volatility': [min_vol_perf[0], min_vol_perf[1], min_vol_perf[2]]
    }, index=['Expected Return', 'Volatility', 'Sharpe Ratio'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Weights Comparison")
        fig_weights = go.Figure()
        for column in portfolio_comparisons.columns:
            fig_weights.add_trace(go.Bar(
                name=column,
                x=portfolio_comparisons.index,
                y=portfolio_comparisons[column] * 100,
                text=(portfolio_comparisons[column] * 100).round(2),
                textposition='auto',
            ))
        fig_weights.update_layout(
            barmode='group',
            title="Portfolio Weights Comparison (%)",
            yaxis_title="Weight (%)",
            height=500
        )
        st.plotly_chart(fig_weights)
    
    with col2:
        st.subheader("Portfolio Metrics Comparison")
        formatted_metrics = metrics_comparison.copy()
        formatted_metrics.loc['Expected Return'] *= 100
        formatted_metrics.loc['Volatility'] *= 100
        
        st.dataframe(formatted_metrics.style.format({
            'Current Portfolio': '{:.2f}',
            'Maximum Sharpe Ratio': '{:.2f}',
            'Minimum Volatility': '{:.2f}'
        }))
        
        fig_metrics = go.Figure()
        for column in formatted_metrics.columns:
            fig_metrics.add_trace(go.Scatter(
                x=['Expected Return', 'Volatility'],
                y=formatted_metrics.loc[['Expected Return', 'Volatility'], column],
                name=column,
                mode='lines+markers'
            ))

        fig_metrics.update_layout(
            title="Risk-Return Comparison",
            yaxis_title="Percentage (%)",
            height=400
        )
        st.plotly_chart(fig_metrics)


        st.write("""
        ### Risk vs. Reward - Explained:

        📊 **Risk** is how much your investment moves up and down. Higher means more uncertainty (big swings in price).
                 
        💰 **Return** is how much profit you make. The further right you go, the more money you could make.

        The graph shows the **trade-off** between the two:
        - **Riskier investments** (higher up) might **give you higher returns** (far right).
        - **Safer investments** (lower) might **offer lower returns** (closer to the left).

        Your goal? Find a place on the graph where **risk and reward** are balanced just right for you!
        """)

#<-----------------------------------------------CAPM Analysis--------------------------------------------------------->
def add_analysis_section(df, returns, market_returns, weights, risk_free_rate=0.0):
    st.header("Advanced Analysis")
    

    st.subheader("CAPM Analysis")
    capm_metrics = {}
    for ticker in returns.columns:
        if ticker != '^GSPC':
            capm_metrics[ticker] = calculate_capm_metrics(
                returns[ticker], 
                market_returns, 
                risk_free_rate
            )
    

    
    capm_df = pd.DataFrame(capm_metrics).T
    st.dataframe(capm_df.style.format("{:.2f}"))
    
    st.write("""
    ### What Does CAPM Tell Us?

    CAPM (Capital Asset Pricing Model) helps us figure out whether an investment is **worth the risk**. 

    - **Expected Return**: What we expect to earn. Higher returns come with more risk!
    - **Risk-Free Rate**: This is what you'd earn from something super safe, like a bond—**guaranteed money**.
    - **Beta**: Measures how much a stock **dances** with the market. A high Beta means it’s a **wild dancer**, jumping higher (and falling harder), while a low Beta means it’s a **calmer dancer**.

    ### Simple Summary:
    - **More Risk = More Reward** — But, is it worth it? CAPM helps us decide.
    - **Beta** shows how risky a stock is: high Beta = more risk (more exciting), low Beta = less risk (more stable).



    """)

    st.subheader("Efficient Frontier")
    portfolio_analysis = PortfolioAnalysis(returns, risk_free_rate)
    efficient_frontier = portfolio_analysis.get_efficient_frontier()

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=efficient_frontier['Volatility'],
        y=efficient_frontier['Return'],
        mode='markers',  
        name='Efficient Frontier',
        marker=dict(
            color=efficient_frontier['Return'],  
            colorscale='Viridis',  
            size=10  
        )
    ))


    fig.update_layout(
        title='Efficient Frontier with Return Color Coding',
        xaxis_title='Volatility (Risk)',
        yaxis_title='Return',
        showlegend=True
    )


            
    

    current_weights = np.array(list(weights.values()))
    current_return, current_vol = portfolio_analysis.calculate_portfolio_metrics(current_weights)
    fig.add_trace(go.Scatter(
        x=[current_vol],
        y=[current_return],
        mode='markers',
        name='Current Portfolio',
        marker=dict(size=10, color='red')
    ))
    
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annual Volatility",
        yaxis_title="Annual Expected Return",
        height=600
    )
    st.plotly_chart(fig)
#<-------------------------------------------------------------------------Financial Ratio analysis---------------------------------->

    st.subheader("Financial Ratios Analysis")
    tabs = st.tabs(["Annual","Quarterly"])
    
    with tabs[1]:
        st.subheader("Quarterly Financial Ratios")
        for ticker in returns.columns:
            if ticker != '^GSPC':
                st.write(f"### {ticker}")
                ratios = calculate_financial_ratios(ticker, period="quarterly")
                if not ratios.empty:
                    st.dataframe(ratios.style.format("{:.2f}"))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ratios.index, y=ratios['Net Profit Margin (%)'], 
                                           name='Net Profit Margin', mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=ratios.index, y=ratios['Return on Equity (%)'], 
                                           name='ROE', mode='lines+markers'))
                    fig.update_layout(title=f"{ticker} Financial Ratios Trends",
                                    xaxis_title="Period",
                                    yaxis_title="Percentage (%)")
                    st.plotly_chart(fig)
                else:
                    st.warning(f"No quarterly financial data available for {ticker}")
    
    with tabs[0]:
        st.subheader("Annual Financial Ratios")
        for ticker in returns.columns:
            if ticker != '^GSPC':
                st.write(f"### {ticker}")
                ratios = calculate_financial_ratios(ticker, period="yearly")
                if not ratios.empty:
                    st.dataframe(ratios.style.format("{:.2f}"))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ratios.index, y=ratios['Net Profit Margin (%)'], 
                                           name='Net Profit Margin', mode='lines+markers'))
                    fig.add_trace(go.Scatter(x=ratios.index, y=ratios['Return on Equity (%)'], 
                                           name='ROE', mode='lines+markers'))
                    fig.update_layout(title=f"{ticker} Financial Ratios Trends",
                                    xaxis_title="Period",
                                    yaxis_title="Percentage (%)")
                    st.plotly_chart(fig)
                else:
                    st.warning(f"No annual financial data available for {ticker}")



#<---------------------------------------------------Main Function ----------------------------------------------------->
def main():
    st.set_page_config(page_title="Financial Risk Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")
    st.title("Financial Risk Analysis Dashboard")
    

    st.sidebar.title("Portfolio Configuration")
    

    default_end_date = date.today()
    default_start_date = default_end_date - timedelta(days=365)
    
    start_date = st.sidebar.date_input("Start Date", value=default_start_date, max_value=default_end_date)
    end_date = st.sidebar.date_input("End Date", value=default_end_date, min_value=start_date, max_value=default_end_date)
    

    stock_input = st.sidebar.text_area(
        "Enter Stock Tickers (one per line)",
        placeholder="e.g.,\nAAPL\nMSFT\nGOOG",
        help="Enter stock tickers (one per line). Example: AAPL for Apple Inc."
    )

    # Initialize selected stocks in session state
    if 'selected_stocks' not in st.session_state:
        st.session_state.selected_stocks = {}

    # Submit Button for Stock Tickers
    if st.sidebar.button("Submit Tickers") and stock_input:
        tickers = [tick.strip().upper() for tick in stock_input.split('\n') if tick.strip()]
        for ticker in tickers:
            if ticker not in st.session_state.selected_stocks:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    company_name = info.get('longName', ticker)
                    st.session_state.selected_stocks[ticker] = company_name
                except Exception as e:
                    st.sidebar.warning(f"Warning: Could not verify {ticker}. Please ensure it's a valid ticker.")

    # Delete stock option
    delete_stock = st.sidebar.selectbox("Delete a stock", ["Select a stock to delete"] + list(st.session_state.selected_stocks.keys()))
    if delete_stock != "Select a stock to delete":
        if st.sidebar.button(f"Delete {delete_stock}"):
            del st.session_state.selected_stocks[delete_stock]
            st.sidebar.success(f"{delete_stock} has been removed.")

    # Investment Amount Input (Placed Above Weights Section)
    investment_amount = st.sidebar.number_input("Total Investment Amount ($)", min_value=0.0, value=10000.0, step=500.0)

    # Risk-Free Rate Input
    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1) / 100.0  # Convert percentage to decimal

    # Portfolio Weights Section
    weights = {}
    total_weight = 0
    
    if len(st.session_state.selected_stocks) > 0:
        st.sidebar.subheader("Portfolio Weights")
        
        # Equal weight button
        if st.sidebar.button("Set Equal Weights"):
            equal_weight = 100.0 / len(st.session_state.selected_stocks)
            for ticker in st.session_state.selected_stocks:
                weights[ticker] = equal_weight / 100
                st.session_state[f"weight_{ticker}"] = equal_weight
        
        # Individual weight inputs for each stock
        for ticker, name in st.session_state.selected_stocks.items():
            default_weight = st.session_state.get(f"weight_{ticker}", 100.0 / len(st.session_state.selected_stocks))
            weight = st.sidebar.number_input(
                f"{name} ({ticker}) Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_weight,
                step=1.0,
                key=f"weight_{ticker}"
            )
            weights[ticker] = weight / 100  # Convert to decimal
            total_weight += weight
        
        if total_weight != 100:
            st.sidebar.warning(f"Total weight = {total_weight}%. Please adjust to 100%")
    
        # Market Index Selection
        index_option = st.sidebar.selectbox(
            "Select Market Index",
            ["^GSPC", "^DJI", "^IXIC"],
            format_func=lambda x: {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}[x]
        )
        
        # Main Analysis Section
        if len(st.session_state.selected_stocks) > 0:
            all_tickers = list(st.session_state.selected_stocks.keys()) + [index_option]
            with st.spinner("Fetching stock data..."):
                df = fetch_data(all_tickers, start_date, end_date)
                if not df.empty:
                    returns = df.pct_change().dropna()
                    market_returns = returns[index_option]

                    # Exclude index returns for portfolio risk calculation
                    stock_returns = returns.drop(columns=[index_option])

                    # Create tabs for Stock Price Visualization and the rest of the analysis
                    tab1, tab2 = st.tabs(["Portfolio Overview", "Stock Price Visualization"])

                    with tab1:
                        # Portfolio Summary
                        st.subheader("Portfolio Summary")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.header("Current Portfolio Allocation")
                            current_allocation = pd.DataFrame({
                                'Asset': [f"{name} ({ticker})" for ticker, name in st.session_state.selected_stocks.items()],
                                'Weight': list(weights.values())
                            })
                            fig_current = px.pie(current_allocation, values='Weight', names='Asset', 
                                                title='Current Weights')
                            st.plotly_chart(fig_current)

                        with col2:
                            st.header("Portfolio Performance Metrics")
                            performance_metrics = calculate_performance_metrics(returns, market_returns)

                            # Display the performance metrics as a table
                            st.dataframe(performance_metrics.style.format("{:.2%}"))

                            # Short descriptions for each metric
                            st.write("""
### Key Metrics Explained:

📈 **Annual Return**: Expected yearly profit. Higher means better performance.

💨 **Annual Volatility**: Measures price swings. More swings = higher risk.

⚖️ **Sharpe Ratio**: Return for each unit of risk. Higher is better.

🚨 **Max Drawdown**: The largest loss from the peak. Lower is better.

📊 **Beta**: Sensitivity to market changes. Higher means more fluctuation compared to the market.
""")

                        # Optimal Portfolio Allocation
                        if len(stock_returns.columns) > 1:  # Need at least 2 stocks for optimization
                            display_optimal_allocations(stock_returns, weights)

                        # Advanced Analysis Section
                        add_analysis_section(df, stock_returns, market_returns, weights, risk_free_rate=risk_free_rate)
                        
                    with tab2:
                        # Real-Time Stock Graphs
                        create_stock_graphs(df, auto_refresh=True)
                        
                else:
                    st.error("Unable to fetch data for the selected stocks. Please verify the tickers.")
        else:
            st.info("Please enter at least one valid stock ticker to begin analysis.")
    else:
        st.info("Please enter stock tickers in the sidebar to begin analysis.")


if __name__ == "__main__":
    main()



