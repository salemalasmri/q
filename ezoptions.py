import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math
from math import log, sqrt
import re
import time
from scipy.stats import norm
import threading
from contextlib import contextmanager
from scipy.interpolate import griddata
import numpy as np
import pytz
from datetime import timedelta
import requests
import json


def calculate_heikin_ashi(df):
    """Calculate Heikin Ashi candlestick values."""
    ha_df = pd.DataFrame(index=df.index)
    
    # Calculate Heikin Ashi values
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize HA_Open with first candle's opening price
    ha_df['HA_Open'] = pd.Series(index=df.index)
    ha_df.loc[ha_df.index[0], 'HA_Open'] = df['Open'].iloc[0]
    
    # Calculate subsequent HA_Open values
    for i in range(1, len(df)):
        ha_df.loc[ha_df.index[i], 'HA_Open'] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2
    
    ha_df['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    return ha_df


def calculate_technical_indicators(df):
    """Calculate various technical indicators for intraday data."""
    if df is None or len(df) == 0:
        return {}
    
    indicators = {}
    selected_indicators = st.session_state.get('selected_indicators', [])
    
    # EMA calculations
    if "EMA (Exponential Moving Average)" in selected_indicators and st.session_state.get('ema_periods'):
        indicators['ema'] = {}
        for period in st.session_state.ema_periods:
            if len(df) >= period:
                indicators['ema'][period] = df['Close'].ewm(span=period, adjust=False).mean()
    
    # SMA calculations
    if "SMA (Simple Moving Average)" in selected_indicators and st.session_state.get('sma_periods'):
        indicators['sma'] = {}
        for period in st.session_state.sma_periods:
            if len(df) >= period:
                indicators['sma'][period] = df['Close'].rolling(window=period).mean()
    
    # Bollinger Bands
    if "Bollinger Bands" in selected_indicators and st.session_state.get('bollinger_period'):
        period = st.session_state.bollinger_period
        std_dev = st.session_state.get('bollinger_std', 2.0)
        if len(df) >= period:
            sma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            indicators['bollinger'] = {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
    
    # RSI
    if "RSI (Relative Strength Index)" in selected_indicators and st.session_state.get('rsi_period'):
        period = st.session_state.rsi_period
        if len(df) >= period + 1:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # VWAP
    if "VWAP (Volume Weighted Average Price)" in selected_indicators and 'Volume' in df.columns:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_volume = df['Volume'].cumsum()
        cumulative_price_volume = (typical_price * df['Volume']).cumsum()
        indicators['vwap'] = cumulative_price_volume / cumulative_volume
    
    return indicators


def calculate_fibonacci_levels(df):
    """Calculate Fibonacci retracement levels based on recent high and low."""
    if df is None or len(df) == 0:
        return {}
    
    # Use the full range for fibonacci calculation
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    
    fibonacci_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    levels = {}
    
    for ratio in fibonacci_ratios:
        levels[f"{ratio:.3f}"] = high - (diff * ratio)
    
    return levels


def add_technical_indicators_to_chart(fig, indicators, fibonacci_levels=None):
    """Add technical indicators to the intraday chart."""
    if not indicators:
        return fig
    
    # Color palette for indicators
    colors = ['#FFD700', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#F0E68C']
    color_index = 0
    
    # Add EMAs
    if 'ema' in indicators:
        ema_periods = list(indicators['ema'].keys())
        for i, (period, ema_data) in enumerate(indicators['ema'].items()):
            fig.add_trace(
                go.Scatter(
                    x=ema_data.index,
                    y=ema_data.values,
                    mode='lines',
                    name=f'EMA {period}',
                    line=dict(color=colors[color_index % len(colors)], width=2),
                    opacity=0.8
                ),
                secondary_y=False
            )
            color_index += 1
        
        # Add EMA cloud if we have at least 2 EMAs
        if len(ema_periods) >= 2:
            # Create cloud between first two EMAs (typically fastest and slower)
            ema1 = indicators['ema'][ema_periods[0]]
            ema2 = indicators['ema'][ema_periods[1]]
            
            # Determine which EMA is above/below for coloring
            fig.add_trace(
                go.Scatter(
                    x=ema1.index,
                    y=ema1.values,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),  # Invisible line
                    showlegend=False,
                    hoverinfo='skip'
                ),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=ema2.index,
                    y=ema2.values,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),  # Invisible line
                    fill='tonexty',
                    fillcolor='rgba(100, 150, 200, 0.1)',  # Light blue cloud
                    name=f'EMA Cloud ({ema_periods[0]}-{ema_periods[1]})',
                    showlegend=True,
                    hoverinfo='skip'
                ),
                secondary_y=False
            )
    
    # Add SMAs
    if 'sma' in indicators:
        for period, sma_data in indicators['sma'].items():
            fig.add_trace(
                go.Scatter(
                    x=sma_data.index,
                    y=sma_data.values,
                    mode='lines',
                    name=f'SMA {period}',
                    line=dict(color=colors[color_index % len(colors)], width=2, dash='dash'),
                    opacity=0.8
                ),
                secondary_y=False
            )
            color_index += 1
    
    # Add Bollinger Bands
    if 'bollinger' in indicators:
        bb = indicators['bollinger']
        # Add middle line first
        fig.add_trace(
            go.Scatter(
                x=bb['middle'].index,
                y=bb['middle'].values,
                mode='lines',
                name=f'BB Middle ({st.session_state.bollinger_period})',
                line=dict(color='gray', width=1, dash='dot'),
                opacity=0.7
            ),
            secondary_y=False
        )
        # Upper band
        fig.add_trace(
            go.Scatter(
                x=bb['upper'].index,
                y=bb['upper'].values,
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                showlegend=False
            ),
            secondary_y=False
        )
        # Lower band with fill
        fig.add_trace(
            go.Scatter(
                x=bb['lower'].index,
                y=bb['lower'].values,
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(128, 128, 128, 0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=False
            ),
            secondary_y=False
        )
    
    # Add VWAP
    if 'vwap' in indicators:
        fig.add_trace(
            go.Scatter(
                x=indicators['vwap'].index,
                y=indicators['vwap'].values,
                mode='lines',
                name='VWAP',
                line=dict(color='orange', width=2, dash='dashdot'),
                opacity=0.8
            ),
            secondary_y=False
        )
    
    # Add Fibonacci levels as horizontal lines
    if fibonacci_levels:
        for level_name, level_value in fibonacci_levels.items():
            fig.add_hline(
                y=level_value,
                line_dash="dash",
                line_color="rgba(255, 255, 255, 0.4)",
                annotation_text=f"Fib {level_name}",
                annotation_position="right",
                annotation_font_size=10
            )
    
    return fig


@contextmanager
def st_thread_context():
    """Thread context management for Streamlit"""
    try:
        if not hasattr(threading.current_thread(), '_StreamlitThread__cached_st'):
           
            import warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*missing ScriptRunContext.*')
        yield
    finally:
        pass


with st_thread_context():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Prevent page dimming during reruns
st.markdown("<style>.element-container{opacity:1 !important}</style>", unsafe_allow_html=True)

# Initialize session state for colors if not already set
if 'call_color' not in st.session_state:
    st.session_state.call_color = '#00FF00'  # Default green for calls
if 'put_color' not in st.session_state:
    st.session_state.put_color = '#FF0000'   # Default red for puts
if 'vix_color' not in st.session_state:
    st.session_state.vix_color = '#800080'   # Default purple for VIX

# -------------------------------
# Helper Functions
# -------------------------------
def format_ticker(ticker):
    """Helper function to format tickers for indices"""
    ticker = ticker.upper()
    if ticker == "SPX":
        return "^SPX"
    elif ticker == "NDX":
        return "^NDX"
    elif ticker == "VIX":
        return "^VIX"
    elif ticker == "DJI":
        return "^DJI"
    elif ticker == "RUT":
        return "^RUT"
    return ticker

def check_market_status():
    """Check if we're in pre-market, market hours, or post-market"""
    # Get current time in PST for market checks
    pacific = datetime.now(tz=pytz.timezone('US/Pacific'))
    
    # Get local time and timezone
    local = datetime.now()
    local_tz = datetime.now().astimezone().tzinfo
    
    market_message = None
    
    if pacific.hour >= 21 or pacific.hour < 7:
        next_update = pacific.replace(hour=7, minute=0) if pacific.hour < 7 else \
                     (pacific + timedelta(days=1)).replace(hour=7, minute=00)
        time_until = next_update - pacific
        hours, remainder = divmod(time_until.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        # Convert PST update time to local time
        local_next_update = next_update.astimezone(local_tz)
        
        market_message = f"""
        ⚠️ **WAIT FOR NEW DATA**
        - Current time: {local.strftime('%I:%M %p')} {local_tz}
        - New data will be available at approximately {local_next_update.strftime('%I:%M %p')}
        - Time until new data: {hours}h {minutes}m
        """
    return market_message

def get_cache_ttl():
    """Get the cache TTL from session state refresh rate, with a minimum of 10 seconds"""
    return max(float(st.session_state.get('refresh_rate', 10)), 10)

def calculate_strike_range(current_price, percentage=None):
    """Calculate strike range based on percentage of current price"""
    if percentage is None:
        percentage = st.session_state.get('strike_range', 1.0)
    return current_price * (percentage / 100.0)

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def fetch_options_for_date(ticker, date, S=None):
    """Fetch options data for a specific date with caching"""
    print(f"Fetching option chain for {ticker} EXP {date}")
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(date)
        calls = chain.calls
        puts = chain.puts
        
        if not calls.empty:
            calls = calls.copy()
            calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
        if not puts.empty:
            puts = puts.copy()
            puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
            
        return calls, puts
    except Exception as e:
        st.error(f"Error fetching options data: {e}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def fetch_all_options(ticker):
    """Fetch all available options with caching"""
    print(f"Fetching all options for {ticker}")
    try:
        stock = yf.Ticker(ticker)
        all_calls = []
        all_puts = []
        
        for next_exp in stock.options:
            try:
                calls, puts = fetch_options_for_date(ticker, next_exp)
                if not calls.empty:
                    all_calls.append(calls)
                if not puts.empty:
                    all_puts.append(puts)
            except Exception as e:
                st.error(f"Error fetching fallback options data: {e}")
        
        if all_calls:
            combined_calls = pd.concat(all_calls, ignore_index=True)
        else:
            combined_calls = pd.DataFrame()
        if all_puts:
            combined_puts = pd.concat(all_puts, ignore_index=True)
        else:
            combined_puts = pd.DataFrame()
        
        return combined_calls, combined_puts
    except Exception as e:
        st.error(f"Error fetching all options: {e}")
        return pd.DataFrame(), pd.DataFrame()

def clear_page_state():
    """Clear all page-specific content and containers"""
    for key in list(st.session_state.keys()):
        if key.startswith(('container_', 'chart_', 'table_', 'page_')):
            del st.session_state[key]
    
    if 'current_page_container' in st.session_state:
        del st.session_state['current_page_container']
    
    st.empty()

def extract_expiry_from_contract(contract_symbol):
    """
    Extracts the expiration date from an option contract symbol.
    Handles both 6-digit (YYMMDD) and 8-digit (YYYYMMDD) date formats.
    """
    pattern = r'[A-Z]+W?(?P<date>\d{6}|\d{8})[CP]\d+'
    match = re.search(pattern, contract_symbol)
    if match:
        date_str = match.group("date")
        try:
            if len(date_str) == 6:
                # Parse as YYMMDD
                expiry_date = datetime.strptime(date_str, "%y%m%d").date()
            else:
                # Parse as YYYYMMDD
                expiry_date = datetime.strptime(date_str, "%Y%m%d").date()
            return expiry_date
        except ValueError:
            return None
    return None

def add_current_price_line(fig, current_price):
    """
    Adds a dashed white line at the current price to a Plotly figure.
    For horizontal bar charts, adds a horizontal line. For other charts, adds a vertical line.
    """
    if st.session_state.chart_type == 'Horizontal Bar':
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            opacity=0.7
        )
    else:
        fig.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="white",
            opacity=0.7,
            annotation_text=f"{current_price}",
            annotation_position="top",
            annotation=dict(
                font=dict(size=st.session_state.chart_text_size)
            )
        )
    return fig

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def get_screener_data(screener_type):
    """Fetch screener data from Yahoo Finance"""
    try:
        response = yf.screen(screener_type)
        if isinstance(response, dict) and 'quotes' in response:
            data = []
            for quote in response['quotes']:
                # Extract relevant information
                info = {
                    'symbol': quote.get('symbol', ''),
                    'shortName': quote.get('shortName', ''),
                    'regularMarketPrice': quote.get('regularMarketPrice', 0),
                    'regularMarketChangePercent': quote.get('regularMarketChangePercent', 0),
                    'regularMarketVolume': quote.get('regularMarketVolume', 0),
                }
                data.append(info)
            return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching screener data: {e}")
        return pd.DataFrame()

def calculate_annualized_return(data, period='1y'):
    """Calculate annualized return rate for each weekday"""
    # Convert period to days
    period_days = {
        '1y': 365,
        '6mo': 180,
        '3mo': 90,
        '1mo': 30,
    }
    days = period_days.get(period, 365)
    
    # Filter data for selected period using proper indexing
    end_date = data.index.max()
    start_date = end_date - pd.Timedelta(days=days)
    filtered_data = data.loc[start_date:end_date].copy()
    
    # Calculate daily returns
    filtered_data['Returns'] = filtered_data['Close'].pct_change()
    
    # Group by weekday and calculate mean return
    weekday_returns = filtered_data.groupby(filtered_data.index.weekday)['Returns'].mean()
    
    # Annualize returns (252 trading days per year)
    annualized_returns = (1 + weekday_returns) ** 252 - 1
    
    # Map weekday numbers to names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    annualized_returns.index = weekday_names
    
    return annualized_returns * 100  # Convert to percentage

def create_weekday_returns_chart(returns):
    """Create a bar chart of weekday returns"""
    fig = go.Figure()
    
    # Add bars with colors based on return value
    for day, value in returns.items():
        color = st.session_state.call_color if value >= 0 else st.session_state.put_color
        fig.add_trace(go.Bar(
            x=[day],
            y=[value],
            name=day,
            marker_color=color,
            text=[f'{value:.2f}%'],
            textposition='outside'
        ))
    
    # Calculate y-axis range with padding
    y_values = returns.values
    y_max = max(y_values)
    y_min = min(y_values)
    y_range = y_max - y_min
    padding = y_range * 0.2  # 20% padding
    
    fig.update_layout(
        title=dict(
            text='Annualized Return Rate by Weekday',
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Weekday',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Annualized Return (%)',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            range=[y_min - padding, y_max + padding],  # Add padding to y-axis range
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        showlegend=False,
        template="plotly_dark"
    )
    
    # Update axis fonts
    fig.update_xaxes(tickfont=dict(size=st.session_state.chart_text_size))
    
    return fig

def analyze_options_flow(calls_df, puts_df, current_price):
    """Analyze options flow to determine bought vs sold contracts"""
    # Deep copy to avoid modifying originals
    calls = calls_df.copy()
    puts = puts_df.copy()
    
    # Determine if option is likely bought/sold based on trade price vs bid/ask
    # For calls: trades near ask = likely bought, trades near bid = likely sold
    calls['trade_type'] = calls.apply(lambda x: 'bought' if x['lastPrice'] >= (x['bid'] + (x['ask'] - x['bid'])*0.6) else 'sold', axis=1)
    puts['trade_type'] = puts.apply(lambda x: 'bought' if x['lastPrice'] >= (x['bid'] + (x['ask'] - x['bid'])*0.6) else 'sold', axis=1)
    
    # Add ITM/OTM classification
    calls['moneyness'] = calls.apply(lambda x: 'ITM' if x['strike'] <= current_price else 'OTM', axis=1)
    puts['moneyness'] = puts.apply(lambda x: 'ITM' if x['strike'] >= current_price else 'OTM', axis=1)
    
    # Calculate volume-weighted stats
    call_stats = {
        'bought': {
            'volume': calls[calls['trade_type'] == 'bought']['volume'].sum(),
            'premium': (calls[calls['trade_type'] == 'bought']['volume'] * calls[calls['trade_type'] == 'bought']['lastPrice'] * 100).sum()
        },
        'sold': {
            'volume': calls[calls['trade_type'] == 'sold']['volume'].sum(),
            'premium': (calls[calls['trade_type'] == 'sold']['volume'] * calls[calls['trade_type'] == 'sold']['lastPrice'] * 100).sum()
        },
        'OTM': {
            'volume': calls[calls['moneyness'] == 'OTM']['volume'].sum(),
            'premium': (calls[calls['moneyness'] == 'OTM']['volume'] * calls[calls['moneyness'] == 'OTM']['lastPrice'] * 100).sum()
        },
        'ITM': {
            'volume': calls[calls['moneyness'] == 'ITM']['volume'].sum(), 
            'premium': (calls[calls['moneyness'] == 'ITM']['volume'] * calls[calls['moneyness'] == 'ITM']['lastPrice'] * 100).sum()
        }
    }
    
    put_stats = {
        'bought': {
            'volume': puts[puts['trade_type'] == 'bought']['volume'].sum(),
            'premium': (puts[puts['trade_type'] == 'bought']['volume'] * puts[puts['trade_type'] == 'bought']['lastPrice'] * 100).sum()
        },
        'sold': {
            'volume': puts[puts['trade_type'] == 'sold']['volume'].sum(),
            'premium': (puts[puts['trade_type'] == 'sold']['volume'] * puts[puts['trade_type'] == 'sold']['lastPrice'] * 100).sum()
        },
        'OTM': {
            'volume': puts[puts['moneyness'] == 'OTM']['volume'].sum(),
            'premium': (puts[puts['moneyness'] == 'OTM']['volume'] * puts[puts['moneyness'] == 'OTM']['lastPrice'] * 100).sum()
        },
        'ITM': {
            'volume': puts[puts['moneyness'] == 'ITM']['volume'].sum(), 
            'premium': (puts[puts['moneyness'] == 'ITM']['volume'] * puts[puts['moneyness'] == 'ITM']['lastPrice'] * 100).sum()
        }
    }
    
    # Calculate OTM bought/sold breakdown
    otm_calls_bought = calls[(calls['moneyness'] == 'OTM') & (calls['trade_type'] == 'bought')]['volume'].sum()
    otm_calls_sold = calls[(calls['moneyness'] == 'OTM') & (calls['trade_type'] == 'sold')]['volume'].sum()
    otm_puts_bought = puts[(puts['moneyness'] == 'OTM') & (puts['trade_type'] == 'bought')]['volume'].sum()
    otm_puts_sold = puts[(puts['moneyness'] == 'OTM') & (puts['trade_type'] == 'sold')]['volume'].sum()
    
    # Calculate total premium values
    total_call_premium = (calls['volume'] * calls['lastPrice'] * 100).sum()
    total_put_premium = (puts['volume'] * puts['lastPrice'] * 100).sum()
    
    return {
        'calls': call_stats,
        'puts': put_stats,
        'otm_detail': {
            'calls_bought': otm_calls_bought,
            'calls_sold': otm_calls_sold,
            'puts_bought': otm_puts_bought,
            'puts_sold': otm_puts_sold
        },
        'total_premium': {
            'calls': total_call_premium,
            'puts': total_put_premium
        }
    }

def create_option_flow_charts(flow_data, title="Options Flow Analysis"):
    """Create visual charts for options flow analysis"""
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    # Create bar chart for bought vs sold
    fig_flow = go.Figure()
    
    # Calls bought/sold
    fig_flow.add_trace(go.Bar(
        x=['Calls Bought', 'Calls Sold'],
        y=[flow_data['calls']['bought']['volume'], flow_data['calls']['sold']['volume']],
        name='Calls',
        marker_color=call_color
    ))
    
    # Puts bought/sold
    fig_flow.add_trace(go.Bar(
        x=['Puts Bought', 'Puts Sold'],
        y=[flow_data['puts']['bought']['volume'], flow_data['puts']['sold']['volume']],
        name='Puts',
        marker_color=put_color
    ))
    
    fig_flow.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Trade Direction',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Volume',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='relative',
        template="plotly_dark"
    )
    
    # Create OTM/ITM chart
    fig_money = go.Figure()
    
    # Calls OTM/ITM
    fig_money.add_trace(go.Bar(
        x=['OTM Calls', 'ITM Calls'],
        y=[flow_data['calls']['OTM']['volume'], flow_data['calls']['ITM']['volume']],
        name='Calls',
        marker_color=call_color
    ))
    
    # Puts OTM/ITM
    fig_money.add_trace(go.Bar(
        x=['OTM Puts', 'ITM Puts'],
        y=[flow_data['puts']['OTM']['volume'], flow_data['puts']['ITM']['volume']],
        name='Puts',
        marker_color=put_color
    ))
    
    fig_money.update_layout(
        title=dict(
            text="OTM vs ITM Volume",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Moneyness',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Volume',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='relative',
        template="plotly_dark"
    )
    
    # Premium chart (donut)
    premium_labels = ['Call Premium', 'Put Premium']
    premium_values = [flow_data['total_premium']['calls'], flow_data['total_premium']['puts']]
    
    fig_premium = go.Figure(data=[go.Pie(
        labels=premium_labels,
        values=premium_values,
        hole=0.4,
        marker=dict(colors=[call_color, put_color])
    )])
    
    total_premium = flow_data['total_premium']['calls'] + flow_data['total_premium']['puts']
    premium_text = f"${total_premium:,.0f}"
    
    fig_premium.update_layout(
        title=dict(
            text="Total Premium Flow",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        legend=dict(font=dict(size=st.session_state.chart_text_size)),
        annotations=[dict(
            text=premium_text,
            x=0.5, y=0.5,
            font=dict(size=st.session_state.chart_text_size + 4),
            showarrow=False
        )],
        template="plotly_dark"
    )
    
    # OTM Analysis Breakdown (horizontal)
    fig_otm = go.Figure()
    
    # OTM Calls bought/sold
    fig_otm.add_trace(go.Bar(
        y=['OTM Calls'],
        x=[flow_data['otm_detail']['calls_bought']],
        name='Bought',
        orientation='h',
        marker_color='lightgreen',
        offsetgroup=0
    ))
    
    fig_otm.add_trace(go.Bar(
        y=['OTM Calls'],
        x=[flow_data['otm_detail']['calls_sold']],
        name='Sold',
        orientation='h',
        marker_color='darkgreen',
        offsetgroup=1
    ))
    
    # OTM Puts bought/sold
    fig_otm.add_trace(go.Bar(
        y=['OTM Puts'],
        x=[flow_data['otm_detail']['puts_bought']],
        name='Bought',
        orientation='h',
        marker_color='pink',
        offsetgroup=0
    ))
    
    fig_otm.add_trace(go.Bar(
        y=['OTM Puts'],
        x=[flow_data['otm_detail']['puts_sold']],
        name='Sold',
        orientation='h',
        marker_color='darkred',
        offsetgroup=1
    ))
    
    fig_otm.update_layout(
        title=dict(
            text="OTM Options Bought vs Sold",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Volume',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        barmode='relative',
        template="plotly_dark"
    )
    
    return fig_flow, fig_money, fig_premium, fig_otm

def create_option_premium_heatmap(calls_df, puts_df, strikes, expiry_dates, current_price):
    """Create a heatmap showing premium distribution across strikes and expiries"""
    # Initialize data matrices
    call_premium = np.zeros((len(expiry_dates), len(strikes)))
    put_premium = np.zeros((len(expiry_dates), len(strikes)))
    
    # Map strikes and expiry dates to indices
    strike_to_idx = {strike: i for i, strike in enumerate(strikes)}
    expiry_to_idx = {expiry: i for i, expiry in enumerate(expiry_dates)}
    
    # Fill matrices with premium data (volume * price)
    for _, row in calls_df.iterrows():
        if row['strike'] in strike_to_idx and row['expiry_date'] in expiry_to_idx:
            i = expiry_to_idx[row['expiry_date']]
            j = strike_to_idx[row['strike']]
            call_premium[i, j] = row['volume'] * row['lastPrice'] * 100
    
    for _, row in puts_df.iterrows():
        if row['strike'] in strike_to_idx and row['expiry_date'] in expiry_to_idx:
            i = expiry_to_idx[row['expiry_date']]
            j = strike_to_idx[row['strike']]
            put_premium[i, j] = row['volume'] * row['lastPrice'] * 100
    
    # Create heatmaps
    fig_calls = go.Figure(data=go.Heatmap(
        z=call_premium,
        x=strikes,
        y=expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(0,255,0,0.1)'], [1, st.session_state.call_color]],
        hoverongaps=False,
        name="Call Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            titleside="top",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_calls.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_calls.update_layout(
        title=dict(
            text="Call Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_puts = go.Figure(data=go.Heatmap(
        z=put_premium,
        x=strikes,
        y=expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(255,0,0,0.1)'], [1, st.session_state.put_color]],
        hoverongaps=False,
        name="Put Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            titleside="top",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_puts.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_puts.update_layout(
        title=dict(
            text="Put Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    return fig_calls, fig_puts

def create_premium_heatmap(calls_df, puts_df, filtered_strikes, selected_expiry_dates, current_price):
    """Create heatmaps showing premium distribution across strikes and expiration dates."""
    # Initialize data matrices
    call_premium = np.zeros((len(selected_expiry_dates), len(filtered_strikes)))
    put_premium = np.zeros((len(selected_expiry_dates), len(filtered_strikes)))
    
    # Map strikes and expiry dates to indices
    strike_to_idx = {strike: i for i, strike in enumerate(filtered_strikes)}
    expiry_to_idx = {expiry: i for i, expiry in enumerate(selected_expiry_dates)}
    
    # Fill matrices with premium data (volume * price)
    for _, row in calls_df.iterrows():
        if row['strike'] in filtered_strikes and row['extracted_expiry'].strftime('%Y-%m-%d') in expiry_to_idx:
            strike_idx = strike_to_idx[row['strike']]
            expiry_idx = expiry_to_idx[row['extracted_expiry'].strftime('%Y-%m-%d')]
            call_premium[expiry_idx][strike_idx] += row['volume'] * row['lastPrice'] * 100
    
    for _, row in puts_df.iterrows():
        if row['strike'] in filtered_strikes and row['extracted_expiry'].strftime('%Y-%m-%d') in expiry_to_idx:
            strike_idx = strike_to_idx[row['strike']]
            expiry_idx = expiry_to_idx[row['extracted_expiry'].strftime('%Y-%m-%d')]
            put_premium[expiry_idx][strike_idx] += row['volume'] * row['lastPrice'] * 100
    
    # Create heatmaps
    fig_calls = go.Figure(data=go.Heatmap(
        z=call_premium,
        x=filtered_strikes,
        y=selected_expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(0,255,0,0.1)'], [1, st.session_state.call_color]],
        hoverongaps=False,
        name="Call Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_calls.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_calls.update_layout(
        title=dict(
            text="Call Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    fig_puts = go.Figure(data=go.Heatmap(
        z=put_premium,
        x=filtered_strikes,
        y=selected_expiry_dates,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.01, 'rgba(255,0,0,0.1)'], [1, st.session_state.put_color]],
        hoverongaps=False,
        name="Put Premium",
        showscale=True,
        colorbar=dict(
            title="Premium ($)",
            tickformat="$,.0f"
        )
    ))
    
    # Add current price line
    fig_puts.add_vline(
        x=current_price,
        line_dash="dash",
        line_color="white",
        opacity=0.7
    )
    
    fig_puts.update_layout(
        title=dict(
            text="Put Premium Heatmap",
            x=0,
            xanchor='left',
            font=dict(size=st.session_state.chart_text_size + 8)
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Expiration Date',
            font=dict(size=st.session_state.chart_text_size)
        ),
        template="plotly_dark",
        yaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        xaxis=dict(
            tickfont=dict(size=st.session_state.chart_text_size)
        )
    )
    
    return fig_calls, fig_puts

# Removed: def create_premium_ratio_chart(calls_df, puts_df): function is deleted

# -------------------------------
# Fetch all options experations and add extract expiry
# -------------------------------
def fetch_all_options(ticker):
    """
    Fetches option chains for all available expirations for the given ticker.
    Returns two DataFrames: one for calls and one for puts, with an added column 'extracted_expiry'.
    """
    print(f"Fetching avaiable expirations for {ticker}")  # Add print statement
    stock = yf.Ticker(ticker)
    all_calls = []
    all_puts = []
    
    if stock.options:
        # Get current market date
        current_market_date = datetime.now().date()
        
        for exp in stock.options:
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                
                # Only process options that haven't expired
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                if exp_date >= current_market_date:
                    if not calls.empty:
                        calls = calls.copy()
                        calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                        all_calls.append(calls)
                    if not puts.empty:
                        puts = puts.copy()
                        puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                        all_puts.append(puts)
            except Exception as e:
                st.error(f"Error fetching chain for expiry {exp}: {e}")
                continue
    else:
        try:
            # Get next valid expiration
            next_exp = stock.options[0] if stock.options else None
            if next_exp:
                chain = stock.option_chain(next_exp)
                calls = chain.calls
                puts = chain.puts
                if not calls.empty:
                    calls = calls.copy()
                    calls['extracted_expiry'] = calls['contractSymbol'].apply(extract_expiry_from_contract)
                    all_calls.append(calls)
                if not puts.empty:
                    puts = puts.copy()
                    puts['extracted_expiry'] = puts['contractSymbol'].apply(extract_expiry_from_contract)
                    all_puts.append(puts)
        except Exception as e:
            st.error(f"Error fetching fallback options data: {e}")
    
    if all_calls:
        combined_calls = pd.concat(all_calls, ignore_index=True)
    else:
        combined_calls = pd.DataFrame()
    if all_puts:
        combined_puts = pd.concat(all_puts, ignore_index=True)
    else:
        combined_puts = pd.DataFrame()
    
    return combined_calls, combined_puts

# Charts and price fetching
@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def get_current_price(ticker):
    """Get current price with fallback logic"""
    print(f"Fetching current price for {ticker}")
    formatted_ticker = ticker.replace('%5E', '^')
    
    if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
        try:
            gspc = yf.Ticker('^GSPC')
            price = gspc.info.get("regularMarketPrice")
            if price is None:
                price = gspc.fast_info.get("lastPrice")
            if price is not None:
                return round(float(price), 2)
        except Exception as e:
            print(f"Error fetching SPX price: {str(e)}")
    
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice")
        if price is None:
            price = stock.fast_info.get("lastPrice")
        if price is not None:
            return round(float(price), 2)
    except Exception as e:
        print(f"Yahoo Finance error: {str(e)}")
    
    return None

def create_oi_volume_charts(calls, puts, S):
    if S is None:
        st.error("Could not fetch underlying price.")
        return

    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create separate dataframes for OI and Volume, filtering out zeros
    calls_oi_df = calls_filtered[['strike', 'openInterest']].copy()
    calls_oi_df = calls_oi_df[calls_oi_df['openInterest'] > 0]  # Changed from != 0 to > 0
    calls_oi_df['OptionType'] = 'Call'
    
    puts_oi_df = puts_filtered[['strike', 'openInterest']].copy()
    puts_oi_df = puts_oi_df[puts_oi_df['openInterest'] > 0]  # Changed from != 0 to > 0
    puts_oi_df['OptionType'] = 'Put'
    
    calls_vol_df = calls_filtered[['strike', 'volume']].copy()
    calls_vol_df = calls_vol_df[calls_vol_df['volume'] > 0]  # Changed from != 0 to > 0
    calls_vol_df['OptionType'] = 'Call'
    
    puts_vol_df = puts_filtered[['strike', 'volume']].copy()
    puts_vol_df = puts_vol_df[puts_vol_df['volume'] > 0]  # Changed from != 0 to > 0
    puts_vol_df['OptionType'] = 'Put'
    
    # Calculate Net Open Interest and Net Volume using filtered data
    net_oi = calls_filtered.groupby('strike')['openInterest'].sum() - puts_filtered.groupby('strike')['openInterest'].sum()
    net_volume = calls_filtered.groupby('strike')['volume'].sum() - puts_filtered.groupby('strike')['volume'].sum()
    
    # Calculate total values for titles (handle empty dataframes)
    total_call_oi = calls_oi_df['openInterest'].sum() if not calls_oi_df.empty else 0
    total_put_oi = puts_oi_df['openInterest'].sum() if not puts_oi_df.empty else 0
    total_call_volume = calls_vol_df['volume'].sum() if not calls_vol_df.empty else 0
    total_put_volume = puts_vol_df['volume'].sum() if not puts_vol_df.empty else 0
    
    # Create titles with totals using HTML for colored values
    oi_title_with_totals = (
        f"Open Interest by Strike     "
        f"<span style='color: {call_color}'>{total_call_oi:,.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_oi:,.0f}</span>"
    )
    
    volume_title_with_totals = (
        f"Volume by Strike     "
        f"<span style='color: {call_color}'>{total_call_volume:,.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_volume:,.0f}</span>"
    )
    
    # Create Open Interest Chart
    fig_oi = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_oi_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=calls_oi_df['strike'],
                x=calls_oi_df['openInterest'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_oi.add_trace(go.Scatter(
                x=calls_oi_df['strike'],
                y=calls_oi_df['openInterest'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_oi_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=puts_oi_df['strike'],
                x=puts_oi_df['openInterest'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_oi.add_trace(go.Scatter(
                x=puts_oi_df['strike'],
                y=puts_oi_df['openInterest'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net OI if enabled
    if st.session_state.show_net and not net_oi.empty:
        if st.session_state.chart_type == 'Bar':
            fig_oi.add_trace(go.Bar(
                x=net_oi.index,
                y=net_oi.values,
                name='Net OI',
                marker_color=[call_color if val >= 0 else put_color for val in net_oi.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_oi.add_trace(go.Bar(
                y=net_oi.index,
                x=net_oi.values,
                name='Net OI',
                marker_color=[call_color if val >= 0 else put_color for val in net_oi.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_oi.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[positive_mask],
                    y=net_oi.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net OI (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[~positive_mask],
                    y=net_oi.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net OI (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_oi.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[positive_mask],
                    y=net_oi.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net OI (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_oi.add_trace(go.Scatter(
                    x=net_oi.index[~positive_mask],
                    y=net_oi.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net OI (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding for OI chart
    y_values = []
    for trace in fig_oi.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        oi_y_min = min(min(y_values), 0)  # Include 0 in the range
        oi_y_max = max(y_values)
        oi_y_range = oi_y_max - oi_y_min
        
        # Add 15% padding on top and 5% on bottom
        oi_padding_top = oi_y_range * 0.15
        oi_padding_bottom = oi_y_range * 0.05
        oi_y_min = oi_y_min - oi_padding_bottom
        oi_y_max = oi_y_max + oi_padding_top
    else:
        # Default values if no valid y values
        oi_y_min = 0
        oi_y_max = 100
    
    # Add padding for x-axis range
    padding = strike_range * 0.1
    
    # Update OI chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_oi.update_layout(
            title=dict(
                text=oi_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Open Interest',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    else:
        fig_oi.update_layout(
            title=dict(
                text=oi_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Open Interest',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    
    # Create Volume Chart
    fig_volume = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=calls_vol_df['strike'],
                x=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=puts_vol_df['strike'],
                x=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net Volume if enabled
    if st.session_state.show_net and not net_volume.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=net_volume.index,
                y=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=net_volume.index,
                x=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding for volume chart
    y_values = []
    for trace in fig_volume.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        vol_y_min = min(min(y_values), 0)  # Include 0 in the range
        vol_y_max = max(y_values)
        vol_y_range = vol_y_max - vol_y_min
        
        # Add 15% padding on top and 5% on bottom
        vol_padding_top = vol_y_range * 0.15
        vol_padding_bottom = vol_y_range * 0.05
        vol_y_min = vol_y_min - vol_padding_bottom
        vol_y_max = vol_y_max + vol_padding_top
    else:
        # Default values if no valid y values
        vol_y_min = 0
        vol_y_max = 100
    
    # Update Volume chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    else:
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600  # Increased height for better visibility
        )
    
    # Add current price line
    S = round(S, 2)
    fig_oi = add_current_price_line(fig_oi, S)
    fig_volume = add_current_price_line(fig_volume, S)
    
    return fig_oi, fig_volume

def create_volume_by_strike_chart(calls, puts, S):
    """Create a standalone volume by strike chart for the dashboard."""
    if S is None:
        st.error("Could not fetch underlying price.")
        return None

    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
    
    # Create separate dataframes for Volume, filtering out zeros
    calls_vol_df = calls_filtered[['strike', 'volume']].copy()
    calls_vol_df = calls_vol_df[calls_vol_df['volume'] > 0]
    calls_vol_df['OptionType'] = 'Call'
    
    puts_vol_df = puts_filtered[['strike', 'volume']].copy()
    puts_vol_df = puts_vol_df[puts_vol_df['volume'] > 0]
    puts_vol_df['OptionType'] = 'Put'
    
    # Calculate Net Volume using filtered data
    net_volume = calls_filtered.groupby('strike')['volume'].sum() - puts_filtered.groupby('strike')['volume'].sum()
    
    # Calculate total values for title (handle empty dataframes)
    total_call_volume = calls_vol_df['volume'].sum() if not calls_vol_df.empty else 0
    total_put_volume = puts_vol_df['volume'].sum() if not puts_vol_df.empty else 0
    
    # Create title with totals using HTML for colored values
    volume_title_with_totals = (
        f"Volume by Strike     "
        f"<span style='color: {call_color}'>{total_call_volume:,.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_volume:,.0f}</span>"
    )
    
    # Create Volume Chart
    fig_volume = go.Figure()
    
    # Add calls if enabled and data exists
    if st.session_state.show_calls and not calls_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=calls_vol_df['strike'],
                x=calls_vol_df['volume'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=calls_vol_df['strike'],
                y=calls_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled and data exists
    if st.session_state.show_puts and not puts_vol_df.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=puts_vol_df['strike'],
                x=puts_vol_df['volume'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig_volume.add_trace(go.Scatter(
                x=puts_vol_df['strike'],
                y=puts_vol_df['volume'],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net Volume if enabled
    if st.session_state.show_net and not net_volume.empty:
        if st.session_state.chart_type == 'Bar':
            fig_volume.add_trace(go.Bar(
                x=net_volume.index,
                y=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig_volume.add_trace(go.Bar(
                y=net_volume.index,
                x=net_volume.values,
                name='Net Volume',
                marker_color=[call_color if val >= 0 else put_color for val in net_volume.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net Volume (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_volume.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[positive_mask],
                    y=net_volume.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig_volume.add_trace(go.Scatter(
                    x=net_volume.index[~positive_mask],
                    y=net_volume.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net Volume (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Update Volume chart layout
    if st.session_state.chart_type == 'Horizontal Bar':
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600
        )
    else:
        fig_volume.update_layout(
            title=dict(
                text=volume_title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Volume',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            template="plotly_dark",
            height=600
        )
    
    # Add current price line
    S = round(S, 2)
    fig_volume = add_current_price_line(fig_volume, S)
    
    return fig_volume

def create_donut_chart(call_volume, put_volume):
    labels = ['Calls', 'Puts']
    values = [call_volume, put_volume]
    # Get colors directly from session state at creation time
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(
        title_text='Call vs Put Volume Ratio',
        title_font_size=st.session_state.chart_text_size + 8,  # Title slightly larger
        showlegend=True,
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        )
    )
    fig.update_traces(
        hoverinfo='label+percent+value',
        marker=dict(colors=[call_color, put_color]),
        textfont=dict(size=st.session_state.chart_text_size)
    )
    return fig

# Greek Calculations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_risk_free_rate():
    """Fetch the current risk-free rate from the 3-month Treasury Bill yield with caching."""
    try:
        # Get current price for the 3-month Treasury Bill
        irx_rate = get_current_price("^IRX")
        
        if irx_rate is not None:
            # Convert percentage to decimal (e.g., 5.2% to 0.052)
            risk_free_rate = irx_rate / 100
        else:
            # Fallback to a default value if price fetch fails
            risk_free_rate = 0.02  # 2% as fallback
            print("Using fallback risk-free rate of 2%")
            
        return risk_free_rate
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        return 0.02  # 2% as fallback

# Initialize risk-free rate in session state if not already present
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = get_risk_free_rate()

def calculate_greeks(flag, S, K, t, sigma):
    """
    Calculate delta, gamma and vanna for an option using Black-Scholes model.
    t: time to expiration in years.
    flag: 'c' for call, 'p' for put.
    """
    try:
        # Add a small offset to prevent division by zero
        t = max(t, 1/1440)  # Minimum 1 minute expressed in years
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        # Calculate delta
        if flag == 'c':
            delta_val = norm.cdf(d1)
        else:  # put
            delta_val = norm.cdf(d1) - 1
        
        # Calculate gamma
        gamma_val = norm.pdf(d1) / (S * sigma * sqrt(t))
        
        # Calculate vega
        vega_val = S * norm.pdf(d1) * sqrt(t)
        
        # Calculate vanna
        vanna_val = -norm.pdf(d1) * d2 / sigma
        
        return delta_val, gamma_val, vanna_val
    except Exception as e:
        st.error(f"Error calculating greeks: {e}")
        return None, None, None

def calculate_charm(flag, S, K, t, sigma):
    """
    Calculate charm (dDelta/dTime) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        norm_d1 = norm.pdf(d1)
        
        if flag == 'c':
            charm = -norm_d1 * (2*r*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t))
        else:  # put
            charm = -norm_d1 * (2*r*t - d2*sigma*sqrt(t)) / (2*t*sigma*sqrt(t)) - r*norm.cdf(-d2)
        
        return charm
    except Exception as e:
        st.error(f"Error calculating charm: {e}")
        return None

def calculate_speed(flag, S, K, t, sigma):
    """
    Calculate speed (dGamma/dSpot) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        # Calculate gamma manually
        gamma = norm.pdf(d1) / (S * sigma * sqrt(t))
        
        # Calculate speed
        speed = -gamma * (d1/(sigma * sqrt(t)) + 1) / S
        
        return speed
    except Exception as e:
        st.error(f"Error calculating speed: {e}")
        return None

def calculate_vomma(flag, S, K, t, sigma):
    """
    Calculate vomma (dVega/dVol) for an option.
    """
    try:
        t = max(t, 1/1440)
        r = st.session_state.risk_free_rate  # Use cached rate from session state
        
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        
        # Calculate vega manually
        vega = S * norm.pdf(d1) * sqrt(t)
        
        # Calculate vomma
        vomma = vega * (d1 * d2) / sigma
        
        return vomma
    except Exception as e:
        st.error(f"Error calculating vomma: {e}")
        return None

def calculate_implied_move(S, calls_df, puts_df):
    """Calculate implied move based on straddle prices."""
    try:
        # Find ATM strike (closest to current price)
        all_strikes = pd.concat([calls_df['strike'], puts_df['strike']]).unique()
        atm_strike = min(all_strikes, key=lambda x: abs(x - S))
        
        # Get ATM call and put prices
        atm_call = calls_df[calls_df['strike'] == atm_strike]
        atm_put = puts_df[puts_df['strike'] == atm_strike]
        
        if not atm_call.empty and not atm_put.empty:
            call_price = atm_call['lastPrice'].iloc[0] if 'lastPrice' in atm_call.columns else atm_call['ask'].iloc[0]
            put_price = atm_put['lastPrice'].iloc[0] if 'lastPrice' in atm_put.columns else atm_put['ask'].iloc[0]
            
            straddle_price = call_price + put_price
            implied_move_pct = (straddle_price / S) * 100
            implied_move_dollars = straddle_price
            
            return {
                'atm_strike': atm_strike,
                'straddle_price': straddle_price,
                'implied_move_pct': implied_move_pct,
                'implied_move_dollars': implied_move_dollars,
                'upper_range': S + implied_move_dollars,
                'lower_range': S - implied_move_dollars
            }
    except Exception as e:
        print(f"Error calculating implied move: {e}")
    
    return None

def find_probability_strikes(calls_df, puts_df, S, expiry_date, target_prob=0.5):
    """Find strikes where there's exactly target_prob chance of being above/below at expiration."""
    try:
        # Calculate probability distribution first
        prob_df = calculate_probability_distribution(calls_df, puts_df, S, expiry_date)
        
        if prob_df.empty:
            return None
        
        # For 50% probability: find the strike where prob_above ≈ 0.5 (median)
        # For other probabilities: find strikes where prob_above = target_prob and prob_above = 1-target_prob
        
        # Industry standard: Use target_prob directly as delta levels
        # For 16 delta: find strikes where prob_above = 0.16 and prob_above = 0.84
        # For 30 delta: find strikes where prob_above = 0.30 and prob_above = 0.70
        
        # Lower bound: target_prob chance of being above
        lower_prob = target_prob
        # Upper bound: (1-target_prob) chance of being above  
        upper_prob = 1 - target_prob
        
        # Find strike where prob_above = lower_prob (lower bound of confidence interval)
        prob_above_target = prob_df.iloc[(prob_df['prob_above'] - lower_prob).abs().argsort()[:1]]
        strike_above = prob_above_target['strike'].iloc[0] if not prob_above_target.empty else None
        actual_prob_above = prob_above_target['prob_above'].iloc[0] if not prob_above_target.empty else None
        
        # Find strike where prob_above = upper_prob (upper bound of confidence interval)
        prob_below_target = prob_df.iloc[(prob_df['prob_above'] - upper_prob).abs().argsort()[:1]]
        strike_below = prob_below_target['strike'].iloc[0] if not prob_below_target.empty else None
        actual_prob_below = 1 - prob_below_target['prob_above'].iloc[0] if not prob_below_target.empty else None
        
        return {
            'strike_above': strike_above,  # Strike with target_prob chance of being above
            'prob_above': actual_prob_above,
            'strike_below': strike_below,  # Strike with target_prob chance of being below  
            'prob_below': actual_prob_below,
            'target_probability': target_prob
        }
    except Exception as e:
        print(f"Error finding probability strikes: {e}")
        return None

def find_delta_strikes(calls_df, puts_df, target_delta=0.5):
    """Find strikes closest to the target delta (for delta-based analysis)."""
    try:
        # For calls, find strike closest to target delta
        if 'calc_delta' in calls_df.columns:
            call_deltas = calls_df[calls_df['calc_delta'].notna()]
            if not call_deltas.empty:
                call_target = call_deltas.iloc[(call_deltas['calc_delta'] - target_delta).abs().argsort()[:1]]
                call_strike = call_target['strike'].iloc[0] if not call_target.empty else None
                call_delta = call_target['calc_delta'].iloc[0] if not call_target.empty else None
            else:
                call_strike, call_delta = None, None
        else:
            call_strike, call_delta = None, None
        
        # For puts, find strike closest to -target_delta (puts have negative delta)
        if 'calc_delta' in puts_df.columns:
            put_deltas = puts_df[puts_df['calc_delta'].notna()]
            if not put_deltas.empty:
                put_target = put_deltas.iloc[(put_deltas['calc_delta'] - (-target_delta)).abs().argsort()[:1]]
                put_strike = put_target['strike'].iloc[0] if not put_target.empty else None
                put_delta = put_target['calc_delta'].iloc[0] if not put_target.empty else None
            else:
                put_strike, put_delta = None, None
        else:
            put_strike, put_delta = None, None
        
        return {
            'call_strike': call_strike,
            'call_delta': call_delta,
            'put_strike': put_strike,
            'put_delta': put_delta,
            'call_prob_itm': call_delta if call_delta else None,
            'put_prob_itm': abs(put_delta) if put_delta else None
        }
    except Exception as e:
        print(f"Error finding delta strikes: {e}")
        return None

def calculate_probability_distribution(calls_df, puts_df, S, expiry_date):
    """Calculate probability distribution from option prices using risk-neutral probabilities."""
    try:
        # Get all strikes and sort them
        all_strikes = sorted(pd.concat([calls_df['strike'], puts_df['strike']]).unique())
        
        probabilities = []
        strikes_data = []
        
        today = datetime.today().date()
        if isinstance(expiry_date, str):
            expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d").date()
        
        t_days = (expiry_date - today).days
        t = max(t_days / 365.0, 1/365)  # At least 1 day
        r = st.session_state.risk_free_rate
        
        for strike in all_strikes:
            # Get call and put data for this strike
            call_data = calls_df[calls_df['strike'] == strike]
            put_data = puts_df[puts_df['strike'] == strike]
            
            # Prefer using implied volatility to calculate risk-neutral probabilities
            iv = None
            if not call_data.empty and 'impliedVolatility' in call_data.columns:
                iv = call_data['impliedVolatility'].iloc[0]
            elif not put_data.empty and 'impliedVolatility' in put_data.columns:
                iv = put_data['impliedVolatility'].iloc[0]
            
            if iv and iv > 0:
                # Calculate risk-neutral probability using Black-Scholes
                try:
                    d1 = (log(S / strike) + (r + 0.5 * iv**2) * t) / (iv * sqrt(t))
                    d2 = d1 - iv * sqrt(t)
                    
                    # Risk-neutral probability of finishing above strike
                    prob_above = norm.cdf(d2)  # Use d2 for risk-neutral probability
                    
                    probabilities.append(prob_above)
                    strikes_data.append(strike)
                except:
                    continue
            else:
                # Fallback to delta if available
                if not call_data.empty and 'calc_delta' in call_data.columns:
                    delta = call_data['calc_delta'].iloc[0]
                    prob_above = delta
                elif not put_data.empty and 'calc_delta' in put_data.columns:
                    delta = put_data['calc_delta'].iloc[0]
                    prob_above = 1 + delta  # Put delta is negative
                else:
                    continue
                
                probabilities.append(prob_above)
                strikes_data.append(strike)
        
        if not strikes_data:
            return pd.DataFrame()
            
        # Sort by strike
        prob_df = pd.DataFrame({
            'strike': strikes_data,
            'prob_above': probabilities,
            'prob_below': [1 - p for p in probabilities]
        }).sort_values('strike').reset_index(drop=True)
        
        return prob_df
        
    except Exception as e:
        print(f"Error calculating probability distribution: {e}")
        return pd.DataFrame()

def create_implied_probabilities_chart(prob_df, S, prob_16_data, prob_30_data, implied_move_data):
    """Create simplified implied probabilities visualization focusing on key levels."""
    try:
        # Get colors from session state
        call_color = st.session_state.call_color
        put_color = st.session_state.put_color

        # Create two simple bar charts: one for probability levels and one for expected range
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                'Key Probability Levels',
                'Expected Trading Range'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # First subplot: Key Probability Levels
        prob_levels = []
        
        # Add current price
        prob_levels.append({
            'level': 'Current Price',
            'strike': S,
            'type': 'neutral'
        })
        
                # Add probability levels (delta-based)
        if prob_16_data:
            if prob_16_data['strike_above']:
                prob_levels.append({
                    'level': '16Δ Above (1σ)',
                    'strike': prob_16_data['strike_above'],
                    'type': 'call'
                })
            if prob_16_data['strike_below']:
                prob_levels.append({
                    'level': '16Δ Below (1σ)',
                    'strike': prob_16_data['strike_below'],
                    'type': 'put'
                })
        
        if prob_30_data:
            if prob_30_data['strike_above']:
                prob_levels.append({
                    'level': '30Δ Above',
                    'strike': prob_30_data['strike_above'],
                    'type': 'call'
                })
            if prob_30_data['strike_below']:
                prob_levels.append({
                    'level': '30Δ Below', 
                    'strike': prob_30_data['strike_below'],
                    'type': 'put'
                })
        
        if prob_levels:
            levels_df = pd.DataFrame(prob_levels)
            # Sort by strike price for better visualization
            levels_df = levels_df.sort_values('strike')
            
            fig.add_trace(
                go.Bar(
                    x=levels_df['level'],
                    y=levels_df['strike'],
                    name='Probability Levels',
                    marker_color=[
                        call_color if row['type'] == 'call' 
                        else put_color if row['type'] == 'put'
                        else 'yellow' for _, row in levels_df.iterrows()
                    ],
                    text=[f"${v:.2f}" for v in levels_df['strike']],
                    textposition='auto',
                    textfont=dict(size=st.session_state.chart_text_size)
                ),
                row=1, col=1
            )
        
        # Second subplot: Expected Trading Range
        if implied_move_data:
            range_data = [
                {'level': 'Lower Range', 'value': implied_move_data['lower_range'], 'type': 'put'},
                {'level': 'Current Price', 'value': S, 'type': 'neutral'},
                {'level': 'Upper Range', 'value': implied_move_data['upper_range'], 'type': 'call'}
            ]
            range_df = pd.DataFrame(range_data)
            
            fig.add_trace(
                go.Bar(
                    x=range_df['level'],
                    y=range_df['value'],
                    name='Trading Range',
                    marker_color=[
                        put_color if row['type'] == 'put'
                        else call_color if row['type'] == 'call'
                        else 'yellow' for _, row in range_df.iterrows()
                    ],
                    text=[f"${v:.2f}" for v in range_df['value']],
                    textposition='auto',
                    textfont=dict(size=st.session_state.chart_text_size)
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=500,  # Reduced height since we have fewer elements
            title=dict(
                text="Implied Probabilities Analysis",
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            showlegend=False,  # No need for legend
            template="plotly_dark",
            # Add more vertical space for labels
            margin=dict(t=100, b=50)
        )
        
        # Update subplot titles
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=st.session_state.chart_text_size + 4)
        
        # Update axes
        fig.update_xaxes(
            title_text="Probability Level",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=1
        )
        fig.update_yaxes(
            title_text="Strike Price ($)",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Price Level",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=2
        )
        fig.update_yaxes(
            title_text="Price ($)",
            title_font=dict(size=st.session_state.chart_text_size),
            tickfont=dict(size=st.session_state.chart_text_size),
            row=1, col=2
        )
        
        return fig
    except Exception as e:
        print(f"Error creating implied probabilities chart: {e}")
        return go.Figure()

# Add error handling for fetching the last price to avoid KeyError.
def get_last_price(stock):
    """Helper function to get the last price of the stock."""
    return get_current_price(stock.ticker)

def validate_expiry(expiry_date):
    """Helper function to validate expiration dates"""
    if expiry_date is None:
        return False
    try:
        current_market_date = datetime.now().date()
        # Allow expirations within -1 days
        days_difference = (expiry_date - current_market_date).days
        return days_difference >= -1
    except Exception:
        return False

def is_valid_trading_day(expiry_date, current_date):
    """Helper function to check if expiry is within valid trading window"""
    days_difference = (expiry_date - current_date).days
    return days_difference >= -1

def fetch_and_process_multiple_dates(ticker, expiry_dates, process_func):
    """
    Fetches and processes data for multiple expiration dates.
    
    Args:
        ticker: Stock ticker symbol
        expiry_dates: List of expiration dates
        process_func: Function to process data for each date
        
    Returns:
        Tuple of processed calls and puts DataFrames
    """
    all_calls = []
    all_puts = []
    
    for date in expiry_dates:
        result = process_func(ticker, date)
        if result is not None:
            calls, puts = result
            if not calls.empty:
                calls['expiry_date'] = date  # Add expiry date column
                all_calls.append(calls)
            if not puts.empty:
                puts['expiry_date'] = date  # Add expiry date column
                all_puts.append(puts)
    
    if all_calls and all_puts:
        combined_calls = pd.concat(all_calls, ignore_index=True)
        combined_puts = pd.concat(all_puts, ignore_index=True)
        return combined_calls, combined_puts
    return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def get_combined_intraday_data(ticker):
    """Get intraday data with fallback logic"""
    formatted_ticker = ticker.replace('%5E', '^')
    stock = yf.Ticker(ticker)
    intraday_data = stock.history(period="1d", interval="1m")
    
    # Filter for market hours (9:30 AM to 4:00 PM ET)
    if not intraday_data.empty:
        # Convert timezone to ET
        eastern = pytz.timezone('US/Eastern')
        intraday_data.index = intraday_data.index.tz_convert(eastern)
        market_start = pd.Timestamp(intraday_data.index[0].date()).replace(hour=9, minute=30)
        market_end = pd.Timestamp(intraday_data.index[0].date()).replace(hour=16, minute=0)
        intraday_data = intraday_data.between_time('09:30', '16:00')
    
    if intraday_data.empty:
        return None, None, None
    
    # Get VIX data if overlay is enabled
    vix_data = None
    if st.session_state.show_vix_overlay:
        try:
            vix = yf.Ticker('^VIX')
            vix_intraday = vix.history(period="1d", interval="1m")
            if not vix_intraday.empty:
                # Convert timezone to ET and filter for market hours
                vix_intraday.index = vix_intraday.index.tz_convert(eastern)
                vix_data = vix_intraday.between_time('09:30', '16:00')
                
                # Align VIX data with stock data timeframes
                if not intraday_data.empty and not vix_data.empty:
                    common_times = intraday_data.index.intersection(vix_data.index)
                    if len(common_times) > 0:
                        intraday_data = intraday_data.loc[common_times]
                        vix_data = vix_data.loc[common_times]
                    else:
                        vix_data = None
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            vix_data = None
    
    intraday_data = intraday_data.copy()
    yahoo_last_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else None
    latest_price = yahoo_last_price
    
    # Use ^GSPC for SPX
    if formatted_ticker in ['^SPX'] or ticker in ['%5ESPX', 'SPX']:
        try:
            gspc = yf.Ticker('^GSPC')
            price = gspc.info.get("regularMarketPrice")
            if price is None:
                price = gspc.fast_info.get("lastPrice")
            if price is not None:
                latest_price = round(float(price), 2)
                # Update the last data point with current price
                last_idx = intraday_data.index[-1]
                new_idx = last_idx + pd.Timedelta(minutes=1)  # Add 1 minute to ensure it shows as latest
                new_row = pd.DataFrame({
                    'Open': [latest_price],
                    'High': [latest_price],
                    'Low': [latest_price],
                    'Close': [latest_price],
                    'Volume': [0]
                }, index=[new_idx])
                intraday_data = pd.concat([intraday_data, new_row])
        except Exception as e:
            print(f"Error fetching SPX price: {str(e)}")
            latest_price = yahoo_last_price
    else:
        try:
            price = get_current_price(ticker)  # Use cached get_current_price
            if price is not None:
                latest_price = round(float(price), 2)
                last_idx = intraday_data.index[-1]
                new_idx = last_idx + pd.Timedelta(minutes=1)
                new_row = pd.DataFrame({
                    'Open': [latest_price],
                    'High': [latest_price],
                    'Low': [latest_price],
                    'Close': [latest_price],
                    'Volume': [0]
                }, index=[new_idx])
                intraday_data = pd.concat([intraday_data, new_row])
        except Exception as e:
            print(f"Error updating latest price: {str(e)}")
    
    return intraday_data, latest_price, vix_data

def create_iv_surface(calls_df, puts_df, current_price, selected_dates=None):
    """Create data for IV surface plot with enhanced smoothing and data validation."""
    # Filter by selected dates if provided
    if selected_dates:
        calls_df = calls_df[calls_df['extracted_expiry'].isin(selected_dates)]
        puts_df = puts_df[puts_df['extracted_expiry'].isin(selected_dates)]
    
    # Combine calls and puts and drop rows with NaN values
    options_data = pd.concat([calls_df, puts_df])
    options_data = options_data.dropna(subset=['impliedVolatility', 'strike', 'extracted_expiry'])
    
    if options_data.empty:
        st.warning("No valid options data available for IV surface.")
        return None, None, None
    
    # Calculate moneyness and months to expiration
    options_data['moneyness'] = options_data['strike'].apply(
        lambda x: (x / current_price) * 100
    )
    
    options_data['months'] = options_data['extracted_expiry'].apply(
        lambda x: (x - datetime.now().date()).days / 30.44
    )
    
    # Remove extreme values
    for col in ['impliedVolatility', 'moneyness', 'months']:
        q1 = options_data[col].quantile(0.01)
        q99 = options_data[col].quantile(0.99)
        options_data = options_data[
            (options_data[col] >= q1) & 
            (options_data[col] <= q99)
        ]
    
    if options_data.empty:
        st.warning("No valid data points after filtering.")
        return None, None, None
    
    # Create grid for interpolation
    moneyness_range = np.linspace(85, 115, 200)
    months_range = np.linspace(
        options_data['months'].min(),
        options_data['months'].max(),
        200
    )
    
    # Create meshgrid
    X, Y = np.meshgrid(moneyness_range, months_range)
    
    try:
        # Prepare data for interpolation
        points = options_data[['moneyness', 'months']].values
        values = options_data['impliedVolatility'].values * 100
        
        # Initial interpolation
        Z = griddata(
            points,
            values,
            (X, Y),
            method='linear',  # Start with linear interpolation
            fill_value=np.nan
        )
        
        # Fill remaining NaN values with nearest neighbor interpolation
        mask = np.isnan(Z)
        Z[mask] = griddata(
            points,
            values,
            (X[mask], Y[mask]),
            method='nearest'
        )
        
        # Apply Gaussian smoothing with multiple passes
        if not np.isnan(Z).any():  # Only smooth if we have valid data
            from scipy.ndimage import gaussian_filter
            Z = gaussian_filter(Z, sigma=1.5)
            Z = gaussian_filter(Z, sigma=0.75)
            Z = gaussian_filter(Z, sigma=0.5)
        
        return X, Y, Z
        
    except Exception as e:
        st.error(f"Error creating IV surface: {str(e)}")
        return None, None, None

#Streamlit UI
st.title("Ez Options Stock Data")

# Modify the reset_session_state function to preserve color settings
def reset_session_state():
    """Reset all session state variables except for essential ones"""
    # Keep track of keys we want to preserve
    preserved_keys = {
        'current_page', 
        'initialized', 
        'saved_ticker', 
        'call_color', 
        'put_color',
        'vix_color',
        'show_calls', 
        'show_puts',
        'show_net',
        'strike_range',
        'chart_type',
        'chart_text_size',
        'refresh_rate',
        'intraday_chart_type',
        'candlestick_type',
        'show_vix_overlay',
        'gex_type',
        'show_gex_levels',
        'show_dex_levels',
        'show_technical_indicators',
        'selected_indicators',
        'ema_periods',
        'sma_periods',
        'bollinger_period',
        'bollinger_std',
        'rsi_period',
        'fibonacci_levels',
        'vwap_enabled',
        'use_volume_for_greeks'
    }
    
    # Initialize visibility settings if they don't exist
    if 'show_calls' not in st.session_state:
        st.session_state.show_calls = True
    if 'show_puts' not in st.session_state:
        st.session_state.show_puts = True
    if 'show_net' not in st.session_state:
        st.session_state.show_net = True
    
    preserved_values = {key: st.session_state[key] 
                       for key in preserved_keys 
                       if key in st.session_state}
    
    # Clear everything safely
    for key in list(st.session_state.keys()):
        if key not in preserved_keys:
            try:
                del st.session_state[key]
            except KeyError:
                pass
    
    # Restore preserved values
    for key, value in preserved_values.items():
        st.session_state[key] = value

    # Reset expiry selection keys explicitly
    expiry_selection_keys = [
        'oi_volume_expiry_multi',
        'volume_ratio_expiry_multi',
        'gamma_expiry_multi',
        'vanna_expiry_multi',
        'delta_expiry_multi',
        'charm_expiry_multi',
        'speed_expiry_multi',
        'vomma_expiry_multi',
        'notional_exposure_expiry_multi',
        'max_pain_expiry_multi'
    ]
    for key in expiry_selection_keys:
        if key in st.session_state:
            del st.session_state[key]

# Add near the top with other session state initializations
if 'selected_expiries' not in st.session_state:
    st.session_state.selected_expiries = {}

@st.fragment
def expiry_selector_fragment(page_name, available_dates):
    """Fragment for expiry date selection that properly resets"""
    container = st.empty()
    
    # Initialize session state for this page's selections
    state_key = f"{page_name}_selected_dates"
    widget_key = f"{page_name}_expiry_selector"
    
    # Initialize previous selection state if not exists
    if f"{widget_key}_prev" not in st.session_state:
        st.session_state[f"{widget_key}_prev"] = []
    
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    
    with container:
        # For implied probabilities page, use single select
        if page_name == "Implied Probabilities":
            # Get the first selected date if any, otherwise None
            current_selection = st.session_state[state_key][0] if st.session_state[state_key] else None
            
            selected = [st.selectbox(
                "Select Expiration Date:",
                options=available_dates,
                index=available_dates.index(current_selection) if current_selection in available_dates else 0,
                key=widget_key
            )]
        else:
            # For all other pages, use multiselect
            selected = st.multiselect(
                "Select Expiration Date(s):",
                options=available_dates,
                default=st.session_state[state_key],
                key=widget_key
            )
        
        # Check if selection changed
        if selected != st.session_state[f"{widget_key}_prev"]:
            st.session_state[state_key] = selected
            st.session_state[f"{widget_key}_prev"] = selected.copy()
            if selected:  # Only rerun if there are selections
                st.rerun()
    
    return selected, container

def handle_page_change(new_page):
    """Handle page navigation and state management"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = new_page
        return True
    
    if st.session_state.current_page != new_page:
        # Clear page-specific state
        old_state_key = f"{st.session_state.current_page}_selected_dates"
        if old_state_key in st.session_state:
            del st.session_state[old_state_key]
            
        if 'expiry_selector_container' in st.session_state:
            st.session_state.expiry_selector_container.empty()
        
        st.session_state.current_page = new_page
        reset_session_state()
        st.rerun()
        return True
    
    return False

# Save selected ticker
def save_ticker(ticker):
    st.session_state.saved_ticker = ticker

# Market Maker Functions
def get_latest_business_day():
    """Get the latest business day (Monday-Friday) from 24 hours ago."""
    now = datetime.now()
    twenty_four_hours_ago = now - timedelta(hours=24)
    
    # Start with the date from 24 hours ago
    target_date = twenty_four_hours_ago
    
    # If the date from 24 hours ago is a weekend, find the most recent weekday
    while target_date.weekday() > 4:  # Saturday=5, Sunday=6
        target_date -= timedelta(days=1)
    
    # Always return a business day, even if it's more than 24 hours ago
    # This ensures we get market maker data for the most recent trading day
    return target_date

def should_run_script():
    """Check if script should run (based on whether there's a trading day within 24 hours)."""
    # The script should run if there's a valid trading day within the last 24 hours
    # This will be determined by get_latest_business_day(), so we'll let it run
    # and let that function determine if there's valid data to fetch
    return True

def get_params_for_date(target_date, symbol=None, symbol_type="U"):
    """Get API parameters for a specific date and optional symbol."""
    # Safety check: ensure we never request future dates
    today = datetime.now().date()
    if target_date.date() > today:
        target_date = datetime.now() - timedelta(days=1)  # Use yesterday instead
        while target_date.weekday() > 4:  # Skip weekends
            target_date -= timedelta(days=1)
    
    params = {
        "format": "csv",
        "volumeQueryType": "O",  # Options
        "symbolType": "ALL" if symbol is None else symbol_type,  # ALL for all symbols, O/U for specific
        "reportType": "D",       # Daily report (can be modified to W or M)
        "accountType": "M",      # Market Maker
        "productKind": "ALL",    # All product types (Equity, Index, etc.)
        "porc": "BOTH",          # Calls and Puts
        "reportDate": target_date.strftime("%Y%m%d")  # Date in YYYYMMDD format
    }
    
    # If a specific symbol is provided, add it as a filter parameter
    if symbol:
        params["symbol"] = symbol.upper()
    
    return params

def process_market_maker_data(csv_data):
    """Process market maker CSV data and return summary statistics"""
    try:
        from io import StringIO
        
        # Skip any header lines that are not part of the CSV data
        lines = csv_data.strip().split('\n')
        
        # Find the first line that looks like CSV data (has commas and starts with a number)
        start_index = 0
        for i, line in enumerate(lines):
            if ',' in line and line.strip() and line.strip()[0].isdigit():
                start_index = i
                break
        
        # Reconstruct CSV data from the actual data lines
        csv_lines = lines[start_index:]
        clean_csv = '\n'.join(csv_lines)
        
        # Read CSV with proper column names based on OCC layout
        df = pd.read_csv(StringIO(clean_csv), header=None, names=[
            'Quantity',           # Column 1: Volume
            'Underlying_Symbol',  # Column 2: Underlying symbol  
            'Options_Symbol',     # Column 3: Options symbol
            'Account_Type',       # Column 4: Account type (C/F/M)
            'Call_Put_Indicator', # Column 5: Call/Put indicator (C/P)
            'Exchange',           # Column 6: Exchange
            'Activity_Date',      # Column 7: Activity date
            'Series_Date'         # Column 8: Series/contract date
        ])
        
        if df.empty:
            return None
        
        # Initialize summary data
        summary = {
            'total_volume': 0,
            'call_volume': 0,
            'put_volume': 0,
            'call_percentage': 0,
            'put_percentage': 0,
            'raw_data': df
        }
        
        # Filter for Market Maker data only
        mm_data = df[df['Account_Type'] == 'M']
        
        if mm_data.empty:
            # If no MM data, use all data
            mm_data = df
        
        # Calculate call and put volumes using the Call_Put_Indicator column
        call_data = mm_data[mm_data['Call_Put_Indicator'] == 'C']
        put_data = mm_data[mm_data['Call_Put_Indicator'] == 'P']
        
        summary['call_volume'] = call_data['Quantity'].sum() if not call_data.empty else 0
        summary['put_volume'] = put_data['Quantity'].sum() if not put_data.empty else 0
        summary['total_volume'] = summary['call_volume'] + summary['put_volume']
        
        if summary['total_volume'] > 0:
            summary['call_percentage'] = (summary['call_volume'] / summary['total_volume']) * 100
            summary['put_percentage'] = (summary['put_volume'] / summary['total_volume']) * 100
        

        
        return summary
        
    except Exception as e:
        # Fallback to original logic if the structured approach fails
        try:
            df = pd.read_csv(StringIO(csv_data))
            if df.empty:
                return None
                
            summary = {
                'total_volume': 0,
                'call_volume': 0,
                'put_volume': 0,
                'call_percentage': 0,
                'put_percentage': 0,
                'raw_data': df
            }
            
            # Try to find volume column
            volume_col = None
            for col in df.columns:
                if any(term in col.lower() for term in ['quantity', 'volume', 'vol']):
                    volume_col = col
                    break
            
            if volume_col is None and len(df.select_dtypes(include=[np.number]).columns) > 0:
                volume_col = df.select_dtypes(include=[np.number]).columns[0]
            
            if volume_col:
                # Try to find call/put indicator column
                cp_col = None
                for col in df.columns:
                    if df[col].dtype == 'object':
                        unique_vals = df[col].unique()
                        if any('C' in str(unique_vals)) and any('P' in str(unique_vals)):
                            cp_col = col
                            break
                
                if cp_col:
                    call_data = df[df[cp_col] == 'C']
                    put_data = df[df[cp_col] == 'P']
                    
                    summary['call_volume'] = call_data[volume_col].sum() if not call_data.empty else 0
                    summary['put_volume'] = put_data[volume_col].sum() if not put_data.empty else 0
                    summary['total_volume'] = summary['call_volume'] + summary['put_volume']
                    
                    if summary['total_volume'] > 0:
                        summary['call_percentage'] = (summary['call_volume'] / summary['total_volume']) * 100
                        summary['put_percentage'] = (summary['put_volume'] / summary['total_volume']) * 100
            
            return summary
            
        except Exception:
            return None

def create_market_maker_charts(summary_data):
    """Create charts for market maker data visualization"""
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color
    
    # Ensure we have valid data for charts
    call_vol = max(summary_data['call_volume'], 0)
    put_vol = max(summary_data['put_volume'], 0)
    
    # If both volumes are zero, create placeholder charts
    if call_vol == 0 and put_vol == 0:
        call_vol, put_vol = 1, 1  # Equal placeholder values
    
    # Call/Put Volume Pie Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Calls', 'Puts'],
        values=[call_vol, put_vol],
        marker_colors=[call_color, put_color],
        hole=0.4,
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig_pie.update_layout(
        title=dict(
            text="Market Maker Call/Put Volume Distribution",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        template="plotly_dark",
        height=500,  # Increased height for bigger chart
        showlegend=False,  # Remove legend
        margin=dict(t=80, b=80, l=80, r=80)  # Add margins for better spacing
    )
    
    # Volume Bar Chart
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        x=['Calls', 'Puts'],
        y=[call_vol, put_vol],
        marker_color=[call_color, put_color],
        text=[f"{summary_data['call_volume']:,}", f"{summary_data['put_volume']:,}"],
        textposition='auto',
        name='Volume'
    ))
    
    fig_bar.update_layout(
        title=dict(
            text="Market Maker Volume Breakdown",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis_title=dict(
            text="Option Type",
            font=dict(size=14)
        ),
        yaxis_title=dict(
            text="Volume",
            font=dict(size=14)
        ),
        template="plotly_dark",
        height=500,  # Increased height for bigger chart
        showlegend=False,
        margin=dict(t=80, b=80, l=80, r=80)  # Add margins for better spacing
    )
    
    return fig_pie, fig_bar

def download_volume_csv(symbol=None, symbol_type="U", expiry_date=None):
    """Download market maker volume data from OCC API"""
    BASE_URL = "https://marketdata.theocc.com/volume-query"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Check if we should run the script
    if not should_run_script():
        return None, "Script is currently disabled."
    
    # Clean symbol for OCC API (remove ^ prefix from index symbols)
    clean_symbol = symbol
    if symbol and symbol.startswith('^'):
        clean_symbol = symbol[1:]  # Remove the ^ prefix
    
    # The API report date should be from the latest business day from 24 hours ago
    api_report_date = get_latest_business_day()
    
    # If expiry_date is provided, use it for display purposes
    # The actual API still uses the api_report_date for the reportDate parameter
    display_date = expiry_date if expiry_date else api_report_date
    
    params = get_params_for_date(api_report_date, clean_symbol, symbol_type)
    
    try:
        # Make the GET request to the OCC volume query endpoint
        response = requests.get(BASE_URL, params=params, headers=HEADERS)
        
        # Check if the request was successful
        if response.status_code == 200:
            if expiry_date:
                # Convert expiry_date string to datetime for formatting
                try:
                    if isinstance(expiry_date, str):
                        expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
                        display_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_dt.strftime('%Y-%m-%d (%A)')} (API report date: {api_report_date.strftime('%Y-%m-%d')})"
                    else:
                        display_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date.strftime('%Y-%m-%d (%A)')} (API report date: {api_report_date.strftime('%Y-%m-%d')})"
                except:
                    display_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date} (API report date: {api_report_date.strftime('%Y-%m-%d')})"
            else:
                display_text = f"Market Maker Positioning Data Retrieved Successfully for {api_report_date.strftime('%Y-%m-%d (%A)')}"
            return response.text, display_text
        else:
            # Try previous business days (up to 5 days back)
            for days_back in range(1, 6):  # Try up to 5 days back
                fallback_date = datetime.now() - timedelta(days=days_back)
                
                # Skip weekends
                if fallback_date.weekday() > 4:
                    continue
                
                params_fallback = get_params_for_date(fallback_date, clean_symbol, symbol_type)
                
                try:
                    response_fallback = requests.get(BASE_URL, params=params_fallback, headers=HEADERS)
                    if response_fallback.status_code == 200:
                        if expiry_date:
                            try:
                                if isinstance(expiry_date, str):
                                    expiry_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
                                    fallback_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_dt.strftime('%Y-%m-%d (%A)')} (API fallback date: {fallback_date.strftime('%Y-%m-%d')})"
                                else:
                                    fallback_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date.strftime('%Y-%m-%d (%A)')} (API fallback date: {fallback_date.strftime('%Y-%m-%d')})"
                            except:
                                fallback_text = f"Market Maker Positioning Data Retrieved Successfully for expiry: {expiry_date} (API fallback date: {fallback_date.strftime('%Y-%m-%d')})"
                        else:
                            fallback_text = f"Market Maker Positioning Data Retrieved Successfully for {fallback_date.strftime('%Y-%m-%d (%A)')}"
                        return response_fallback.text, fallback_text
                except requests.exceptions.RequestException:
                    continue
            
            return None, f"Error: Failed to download CSV. Status code: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return None, f"Error during request: {e}"

st.sidebar.title("📊 Navigation")

# Define pages with their corresponding icons
page_icons = {
    "Dashboard": "🏠",
    "OI & Volume": "📈", 
    "Gamma Exposure": "🔺",
    "Delta Exposure": "📊",
    "Vanna Exposure": "🌊",
    "Charm Exposure": "⚡",
    "Speed Exposure": "🚀",
    "Vomma Exposure": "💫",
    "Exposure by Notional Value": "💰",
    "Delta-Adjusted Value Index": "📉",
    "Max Pain": "🎯",
    "GEX Surface": "🗻",
    "IV Surface": "🌐",
    "Implied Probabilities": "🎲",
    "Analysis": "🔍",
    "Calculated Greeks": "🧮"
}

pages = ["Dashboard", "OI & Volume", "Gamma Exposure", "Delta Exposure", 
          "Vanna Exposure", "Charm Exposure", "Speed Exposure", "Vomma Exposure", "Exposure by Notional Value", "Delta-Adjusted Value Index", "Max Pain", "GEX Surface", "IV Surface",
          "Implied Probabilities", "Analysis", "Calculated Greeks"]

# Create page options with icons
page_options = [f"{page_icons[page]} {page}" for page in pages]

# Track the previous page in session state
if 'previous_page' not in st.session_state:
    st.session_state.previous_page = None

selected_page_with_icon = st.sidebar.radio("Select a page:", page_options)

# Extract the actual page name (remove icon and space)
new_page = selected_page_with_icon.split(" ", 1)[1]

# Check if the page has changed
if st.session_state.previous_page != new_page:
    st.session_state.previous_page = new_page
    handle_page_change(new_page)
    # Clear out any page-specific expiry selections
    expiry_selection_keys = [
        'oi_volume_expiry_multi',
        'volume_ratio_expiry_multi',
        'gamma_expiry_multi',
        'vanna_expiry_multi',
        'delta_expiry_multi',
        'charm_expiry_multi',
        'speed_expiry_multi',
        'vomma_expiry_multi',
        'notional_exposure_expiry_multi',
        'max_pain_expiry_multi',
        'implied_probabilities_expiry_multi'
    ]
    for key in expiry_selection_keys:
        if key in st.session_state:
            del st.session_state[key]

# Add after st.sidebar.title("Navigation")
def chart_settings():
    with st.sidebar.expander("Chart Settings", expanded=False):
        # Greek Exposure Settings - FIRST SETTING
        st.write("Greek Exposure Settings:")
        
        # Initialize volume/OI preference setting
        if 'use_volume_for_greeks' not in st.session_state:
            st.session_state.use_volume_for_greeks = False  # Default to Open Interest
        
        use_volume = st.checkbox(
            "Use Volume instead of Open Interest for Greek exposures",
            value=st.session_state.use_volume_for_greeks,
            help="When enabled, Greek exposures (Gamma, Vanna, Delta, Charm, Speed, Vomma) will use Volume instead of Open Interest in calculations"
        )
        
        # Update session state when the setting changes
        if use_volume != st.session_state.use_volume_for_greeks:
            st.session_state.use_volume_for_greeks = use_volume

        st.write("Colors:")
        new_call_color = st.color_picker("Calls", st.session_state.call_color)
        new_put_color = st.color_picker("Puts", st.session_state.put_color)
        
        # Add intraday chart type selection
        if 'intraday_chart_type' not in st.session_state:
            st.session_state.intraday_chart_type = 'Candlestick'
        
        if 'candlestick_type' not in st.session_state:
            st.session_state.candlestick_type = 'Filled'
        
        intraday_type = st.selectbox(
            "Intraday Chart Type:",
            options=['Candlestick', 'Line'],
            index=['Candlestick', 'Line'].index(st.session_state.intraday_chart_type)
        )
        
        # Only show candlestick type selection when candlestick chart is selected
        if intraday_type == 'Candlestick':
            candlestick_type = st.selectbox(
                "Candlestick Style:",
                options=['Filled', 'Hollow', 'Heikin Ashi'],
                index=['Filled', 'Hollow', 'Heikin Ashi'].index(st.session_state.candlestick_type)
            )
            
            if candlestick_type != st.session_state.candlestick_type:
                st.session_state.candlestick_type = candlestick_type
        
        # Update session state when intraday chart type changes
        if intraday_type != st.session_state.intraday_chart_type:
            st.session_state.intraday_chart_type = intraday_type

        if 'show_vix_overlay' not in st.session_state:
            st.session_state.show_vix_overlay = False
        
        # Group VIX settings together
        st.write("VIX Settings:")
        show_vix = st.checkbox("VIX Overlay", value=st.session_state.show_vix_overlay)
        if show_vix:
            new_vix_color = st.color_picker("VIX Color", st.session_state.vix_color)
            if new_vix_color != st.session_state.vix_color:
                st.session_state.vix_color = new_vix_color
        
        if show_vix != st.session_state.show_vix_overlay:
            st.session_state.show_vix_overlay = show_vix

        # Technical Indicators Settings
        st.write("Technical Indicators:")
        
        # Initialize technical indicators settings
        if 'show_technical_indicators' not in st.session_state:
            st.session_state.show_technical_indicators = False
        if 'selected_indicators' not in st.session_state:
            st.session_state.selected_indicators = []
        if 'ema_periods' not in st.session_state:
            st.session_state.ema_periods = [9, 21, 50]
        if 'sma_periods' not in st.session_state:
            st.session_state.sma_periods = [20, 50]
        if 'bollinger_period' not in st.session_state:
            st.session_state.bollinger_period = 20
        if 'bollinger_std' not in st.session_state:
            st.session_state.bollinger_std = 2.0
        if 'rsi_period' not in st.session_state:
            st.session_state.rsi_period = 14
        if 'fibonacci_levels' not in st.session_state:
            st.session_state.fibonacci_levels = True
        if 'vwap_enabled' not in st.session_state:
            st.session_state.vwap_enabled = False
        
        show_technical = st.checkbox("Enable Technical Indicators", value=st.session_state.show_technical_indicators, key="enable_technical_indicators")
        
        # Update technical indicators toggle immediately
        if show_technical != st.session_state.show_technical_indicators:
            st.session_state.show_technical_indicators = show_technical
            # Clear selected indicators when disabling
            if not show_technical:
                st.session_state.selected_indicators = []
        
        if show_technical:
            # Available indicators
            available_indicators = [
                "EMA (Exponential Moving Average)",
                "SMA (Simple Moving Average)", 
                "Bollinger Bands",
                "RSI (Relative Strength Index)",
                "VWAP (Volume Weighted Average Price)",
                "Fibonacci Retracements"
            ]
            
            selected_indicators = st.multiselect(
                "Select Indicators:",
                available_indicators,
                default=st.session_state.selected_indicators,
                key="technical_indicators_selector"
            )
            
            # Update session state immediately when selection changes
            if selected_indicators != st.session_state.selected_indicators:
                st.session_state.selected_indicators = selected_indicators
            
            # EMA Settings
            if "EMA (Exponential Moving Average)" in selected_indicators:
                st.write("**EMA Settings:**")
                ema_input = st.text_input(
                    "EMA Periods (comma-separated)",
                    value=",".join(map(str, st.session_state.ema_periods)),
                    help="e.g., 9,21,50",
                    key="ema_periods_input"
                )
                try:
                    ema_periods = [int(x.strip()) for x in ema_input.split(",") if x.strip()]
                    if ema_periods != st.session_state.ema_periods:
                        st.session_state.ema_periods = ema_periods
                except:
                    st.warning("Invalid EMA periods format. Use comma-separated integers.")
            
            # SMA Settings  
            if "SMA (Simple Moving Average)" in selected_indicators:
                st.write("**SMA Settings:**")
                sma_input = st.text_input(
                    "SMA Periods (comma-separated)",
                    value=",".join(map(str, st.session_state.sma_periods)),
                    help="e.g., 20,50",
                    key="sma_periods_input"
                )
                try:
                    sma_periods = [int(x.strip()) for x in sma_input.split(",") if x.strip()]
                    if sma_periods != st.session_state.sma_periods:
                        st.session_state.sma_periods = sma_periods
                except:
                    st.warning("Invalid SMA periods format. Use comma-separated integers.")
            
            # Bollinger Bands Settings
            if "Bollinger Bands" in selected_indicators:
                st.write("**Bollinger Bands Settings:**")
                bb_period = st.number_input("Period", min_value=5, max_value=50, value=st.session_state.bollinger_period, key="bb_period_input")
                bb_std = st.number_input("Standard Deviations", min_value=1.0, max_value=3.0, value=st.session_state.bollinger_std, step=0.1, key="bb_std_input")
                if bb_period != st.session_state.bollinger_period:
                    st.session_state.bollinger_period = bb_period
                if bb_std != st.session_state.bollinger_std:
                    st.session_state.bollinger_std = bb_std
            
            # RSI Settings
            if "RSI (Relative Strength Index)" in selected_indicators:
                st.write("**RSI Settings:**")
                rsi_period = st.number_input("RSI Period", min_value=5, max_value=30, value=st.session_state.rsi_period, key="rsi_period_input")
                if rsi_period != st.session_state.rsi_period:
                    st.session_state.rsi_period = rsi_period
            
        if 'chart_text_size' not in st.session_state:
            st.session_state.chart_text_size = 15  # Default text size
            
        new_text_size = st.number_input(
            "Chart Text Size",
            min_value=10,
            max_value=30,
            value=st.session_state.chart_text_size,
            step=1,
            help="Adjust the size of text in charts (titles, labels, etc.)"
        )
        
        # Update session state and trigger rerun if text size changes
        if new_text_size != st.session_state.chart_text_size:
            st.session_state.chart_text_size = new_text_size
        
        # Update session state and trigger rerun if either color changes
        if new_call_color != st.session_state.call_color or new_put_color != st.session_state.put_color:
            st.session_state.call_color = new_call_color
            st.session_state.put_color = new_put_color

        st.write("Show/Hide Elements:")
        # Initialize visibility settings if not already set
        if 'show_calls' not in st.session_state:
            st.session_state.show_calls = False
        if 'show_puts' not in st.session_state:
            st.session_state.show_puts = False
        if 'show_net' not in st.session_state:
            st.session_state.show_net = True

        # Visibility toggles
        show_calls = st.checkbox("Show Calls", value=st.session_state.show_calls)
        show_puts = st.checkbox("Show Puts", value=st.session_state.show_puts)
        show_net = st.checkbox("Show Net", value=st.session_state.show_net)

        # Update session state when visibility changes
        if show_calls != st.session_state.show_calls or show_puts != st.session_state.show_puts or show_net != st.session_state.show_net:
            st.session_state.show_calls = show_calls
            st.session_state.show_puts = show_puts
            st.session_state.show_net = show_net

        # Initialize strike range in session state (as percentage)
        if 'strike_range' not in st.session_state:
            st.session_state.strike_range = 2.0  # Default 2%
        
        # Add strike range control (as percentage)
        st.session_state.strike_range = st.number_input(
            "Strike Range (% from current price)",
            min_value=0.1,
            max_value=50.0,
            value=st.session_state.strike_range,
            step=0.1,
            format="%.1f",
            help="Percentage range from current price (e.g., 1.0 = ±1%)",
            key="strike_range_sidebar"
        )

        if 'chart_type' not in st.session_state:
            st.session_state.chart_type = 'Bar'  # Default chart type

        chart_type = st.selectbox(
            "Chart Type:",
            options=['Bar', 'Horizontal Bar', 'Scatter', 'Line', 'Area'],
            index=['Bar', 'Horizontal Bar', 'Scatter', 'Line', 'Area'].index(st.session_state.chart_type)
        )

        # Update session state when chart type changes
        if chart_type != st.session_state.chart_type:
            st.session_state.chart_type = chart_type

         
        if 'gex_type' not in st.session_state:
            st.session_state.gex_type = 'Absolute'  # Default to Absolute GEX
        
        gex_type = st.selectbox(
            "Gamma Exposure Type:",
            options=['Net', 'Absolute'],
            index=['Net', 'Absolute'].index(st.session_state.gex_type)
        )

        # Update session state when GEX type changes
        if gex_type != st.session_state.gex_type:
            st.session_state.gex_type = gex_type

        # Add intraday chart level settings
        st.write("Intraday Chart Levels:")
        
        # Initialize GEX and DEX level settings if not already set
        if 'show_gex_levels' not in st.session_state:
            st.session_state.show_gex_levels = True  # Default to showing GEX levels
        if 'show_dex_levels' not in st.session_state:
            st.session_state.show_dex_levels = False  # Default to not showing DEX levels
        
        # GEX and DEX level toggles
        show_gex_levels = st.checkbox("Show GEX Levels", value=st.session_state.show_gex_levels)
        show_dex_levels = st.checkbox("Show DEX Levels", value=st.session_state.show_dex_levels)
        
        # Update session state when level visibility changes
        if show_gex_levels != st.session_state.show_gex_levels or show_dex_levels != st.session_state.show_dex_levels:
            st.session_state.show_gex_levels = show_gex_levels
            st.session_state.show_dex_levels = show_dex_levels

        # Add refresh rate control before chart type
        if 'refresh_rate' not in st.session_state:
            st.session_state.refresh_rate = 10  # Default refresh rate
        
        new_refresh_rate = st.number_input(
            "Auto-Refresh Rate (seconds)",
            min_value=10,
            max_value=300,
            value=int(st.session_state.refresh_rate),
            step=1,
            help="How often to auto-refresh the page (minimum 10 seconds)"
        )
        
        if new_refresh_rate != st.session_state.refresh_rate:
            print(f"Changing refresh rate from {st.session_state.refresh_rate} to {new_refresh_rate} seconds")
            st.session_state.refresh_rate = float(new_refresh_rate)
            st.cache_data.clear()
            st.rerun()

# Call the regular function instead of the fragment
chart_settings()

# Use the saved ticker and expiry date if available
saved_ticker = st.session_state.get("saved_ticker", "")
saved_expiry_date = st.session_state.get("saved_expiry_date", None)

def validate_expiry(expiry_date):
    """Helper function to validate expiration dates"""
    if expiry_date is None:
        return False
    try:
        current_market_date = datetime.now().date()
        # For future dates, ensure they're treated as valid
        return expiry_date >= current_market_date
    except Exception:
        return False

def compute_greeks_and_charts(ticker, expiry_date_str, page_key, S):
    """Compute greeks and create charts for options data"""
    if not expiry_date_str:
        st.warning("Please select an expiration date.")
        return None, None, None, None, None, None
        
    calls, puts = fetch_options_for_date(ticker, expiry_date_str, S)
    if calls.empty and puts.empty:
        st.warning("No options data available for this ticker.")
        return None, None, None, None, None, None

    combined = pd.concat([calls, puts])
    combined = combined.dropna(subset=['extracted_expiry'])
    selected_expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
    calls = calls[calls['extracted_expiry'] == selected_expiry]
    puts = puts[puts['extracted_expiry'] == selected_expiry]

    # Always use get_current_price to ensure consistent price source
    if S is None:
        st.error("Could not fetch underlying price.")
        return None, None, None, None, None, None

    S = float(S)  # Ensure price is float
    today = datetime.today().date()
    t_days = (selected_expiry - today).days
    if not is_valid_trading_day(selected_expiry, today):
        st.error("The selected expiration date is in the past!")
        return None, None, None, None, None, None

    t = t_days / 365.0

    # Compute Greeks for Gamma, Vanna, Delta, Charm, Speed, and Vomma
    def compute_greeks(row, flag, greek_type):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            delta_val, gamma_val, vanna_val = calculate_greeks(flag, S, row["strike"], t, sigma)
            if greek_type == "gamma":
                return gamma_val
            elif greek_type == "vanna":
                return vanna_val
            elif greek_type == "delta":
                return delta_val
        except Exception:
            return None

    def compute_charm(row, flag):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            charm_val = calculate_charm(flag, S, row["strike"], t, sigma)
            return charm_val
        except Exception:
            return None

    def compute_speed(row, flag):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            speed_val = calculate_speed(flag, S, row["strike"], t, sigma)
            return speed_val
        except Exception:
            return None

    def compute_vomma(row, flag):
        sigma = row.get("impliedVolatility", None)
        if sigma is None or sigma <= 0:
            return None
        try:
            vomma_val = calculate_vomma(flag, S, row["strike"], t, sigma)
            return vomma_val
        except Exception:
            return None

    calls = calls.copy()
    puts = puts.copy()
    calls["calc_gamma"] = calls.apply(lambda row: compute_greeks(row, "c", "gamma"), axis=1)
    puts["calc_gamma"] = puts.apply(lambda row: compute_greeks(row, "p", "gamma"), axis=1)
    calls["calc_vanna"] = calls.apply(lambda row: compute_greeks(row, "c", "vanna"), axis=1)
    puts["calc_vanna"] = puts.apply(lambda row: compute_greeks(row, "p", "vanna"), axis=1)
    calls["calc_delta"] = calls.apply(lambda row: compute_greeks(row, "c", "delta"), axis=1)
    puts["calc_delta"] = puts.apply(lambda row: compute_greeks(row, "p", "delta"), axis=1)
    calls["calc_charm"] = calls.apply(lambda row: compute_charm(row, "c"), axis=1)
    puts["calc_charm"] = puts.apply(lambda row: compute_charm(row, "p"), axis=1)
    calls["calc_speed"] = calls.apply(lambda row: compute_speed(row, "c"), axis=1)
    puts["calc_speed"] = puts.apply(lambda row: compute_speed(row, "p"), axis=1)
    calls["calc_vomma"] = calls.apply(lambda row: compute_vomma(row, "c"), axis=1)
    puts["calc_vomma"] = puts.apply(lambda row: compute_vomma(row, "p"), axis=1)

    calls = calls.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed", "calc_vomma"])
    puts = puts.dropna(subset=["calc_gamma", "calc_vanna", "calc_delta", "calc_charm", "calc_speed", "calc_vomma"])

    # Determine which metric to use based on settings
    volume_metric = 'volume' if st.session_state.get('use_volume_for_greeks', False) else 'openInterest'

    # Correct exposure formulas with proper scaling
    # GEX = Gamma * Volume/OI * Contract Size * Spot Price^2 * 0.01 (for 1% move sensitivity)
    calls["GEX"] = calls["calc_gamma"] * calls[volume_metric] * 100 * S * S * 0.01
    puts["GEX"] = puts["calc_gamma"] * puts[volume_metric] * 100 * S * S * 0.01
    
    # VEX = Vanna * Volume/OI * Contract Size * Spot Price * 0.01 (change in dollar delta per 1% vol change)
    calls["VEX"] = calls["calc_vanna"] * calls[volume_metric] * 100 * S * 0.01
    puts["VEX"] = puts["calc_vanna"] * puts[volume_metric] * 100 * S * 0.01
    
    # DEX = Delta * Volume/OI * Contract Size * Spot Price
    calls["DEX"] = calls["calc_delta"] * calls[volume_metric] * 100 * S
    puts["DEX"] = puts["calc_delta"] * puts[volume_metric] * 100 * S
    
    # Charm = Charm * Volume/OI * Contract Size * Spot Price / 365 (change in dollar delta per day)
    calls["Charm"] = calls["calc_charm"] * calls[volume_metric] * 100 * S / 365.0
    puts["Charm"] = puts["calc_charm"] * puts[volume_metric] * 100 * S / 365.0
    
    # Speed = Total portfolio speed (dGamma/dSpot)
    calls["Speed"] = calls["calc_speed"] * calls[volume_metric] * 100
    puts["Speed"] = puts["calc_speed"] * puts[volume_metric] * 100
    
    # Vomma = Vomma * Volume/OI * Contract Size * 0.01 (change in dollar vega per 1% vol change)
    calls["Vomma"] = calls["calc_vomma"] * calls[volume_metric] * 100 * 0.01
    puts["Vomma"] = puts["calc_vomma"] * puts[volume_metric] * 100 * 0.01

    return calls, puts, S, t, selected_expiry, today

def create_exposure_bar_chart(calls, puts, exposure_type, title, S):
    # Get colors from session state at the start
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Filter out zero values
    calls_df = calls[['strike', exposure_type]].copy()
    calls_df = calls_df[calls_df[exposure_type] != 0]
    calls_df['OptionType'] = 'Call'

    puts_df = puts[['strike', exposure_type]].copy()
    puts_df = puts_df[puts_df[exposure_type] != 0]
    puts_df['OptionType'] = 'Put'

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Apply strike range filter
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    # Filter the original dataframes for net exposure calculation
    calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
    puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

    # Calculate Net Exposure based on type using filtered data
    if exposure_type == 'GEX' or exposure_type == 'GEX_notional':
        if st.session_state.gex_type == 'Net':
            net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() - puts_filtered.groupby('strike')[exposure_type].sum()
        else:  # Absolute
            calls_gex = calls_filtered.groupby('strike')[exposure_type].sum()
            puts_gex = puts_filtered.groupby('strike')[exposure_type].sum()
            net_exposure = pd.Series(index=set(calls_gex.index) | set(puts_gex.index))
            for strike in net_exposure.index:
                call_val = abs(calls_gex.get(strike, 0))
                put_val = abs(puts_gex.get(strike, 0))
                net_exposure[strike] = call_val if call_val >= put_val else -put_val
    elif exposure_type == 'DEX':
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() + puts_filtered.groupby('strike')[exposure_type].sum()
    else:  # VEX, Charm, Speed, Vomma
        net_exposure = calls_filtered.groupby('strike')[exposure_type].sum() + puts_filtered.groupby('strike')[exposure_type].sum()

    # Calculate total Greek values
    total_call_value = calls_df[exposure_type].sum()
    total_put_value = puts_df[exposure_type].sum()

    # Get the metric being used and add it to the title
    metric_name = "Volume" if st.session_state.get('use_volume_for_greeks', False) else "Open Interest"
    
    # Update title to include total Greek values with colored values using HTML and metric info
    title_with_totals = (
        f"{title} ({metric_name})     "
        f"<span style='color: {st.session_state.call_color}'>{total_call_value:,.0f}</span> | "
        f"<span style='color: {st.session_state.put_color}'>{total_put_value:,.0f}</span>"
    )

    fig = go.Figure()

    # Add calls if enabled
    if (st.session_state.show_calls):
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=calls_df['strike'],
                x=calls_df[exposure_type],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df[exposure_type],
                mode='lines',
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled
    if st.session_state.show_puts:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=puts_df['strike'],
                x=puts_df[exposure_type],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df[exposure_type],
                mode='lines',
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net if enabled
    if st.session_state.show_net and not net_exposure.empty:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=net_exposure.index,
                y=net_exposure.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_exposure.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=net_exposure.index,
                x=net_exposure.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_exposure.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_exposure.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[positive_mask],
                    y=net_exposure.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[~positive_mask],
                    y=net_exposure.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_exposure.values >= 0
            
            # Plot positive values
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[positive_mask],
                    y=net_exposure.values[positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            
            # Plot negative values
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_exposure.index[~positive_mask],
                    y=net_exposure.values[~positive_mask],
                    mode='lines',
                    fill='tozeroy',
                    name='Net (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Calculate y-axis range with improved padding
    y_values = []
    for trace in fig.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        y_range = y_max - y_min
        
        # Ensure minimum range and add padding
        if abs(y_range) < 1:
            y_range = 1
        
        # Add 15% padding on top and bottom
        padding = y_range * 0.15
        y_min = y_min - padding
        y_max = y_max + padding
    else:
        # Default values if no valid y values
        y_min = -1
        y_max = 1

    # Update layout with calculated y-range
    padding = strike_range * 0.1
    if st.session_state.chart_type == 'Horizontal Bar':
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                xref="paper",
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)  # Title slightly larger
            ),
            xaxis_title=dict(
                text=f"{title} ({metric_name})",
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='y unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increase chart height for better visibility
        )
    else:
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                xref="paper",
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)  # Title slightly larger
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text=f"{title} ({metric_name})",
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increase chart height for better visibility
        )

    fig = add_current_price_line(fig, S)
    return fig

def calculate_max_pain(calls, puts):
    """Calculate max pain points based on call and put options."""
    if calls.empty or puts.empty:
        return None, None, None, None, None

    unique_strikes = sorted(set(calls['strike'].unique()) | set(puts['strike'].unique()))
    total_pain_by_strike = {}
    call_pain_by_strike = {}
    put_pain_by_strike = {}

    for strike in unique_strikes:
        # Calculate call pain (loss to option writers)
        call_pain = calls[calls['strike'] <= strike]['openInterest'] * (strike - calls[calls['strike'] <= strike]['strike'])
        call_pain_sum = call_pain.sum()
        call_pain_by_strike[strike] = call_pain_sum
        
        # Calculate put pain (loss to option writers)
        put_pain = puts[puts['strike'] >= strike]['openInterest'] * (puts[puts['strike'] >= strike]['strike'] - strike)
        put_pain_sum = put_pain.sum()
        put_pain_by_strike[strike] = put_pain_sum
        
        total_pain_by_strike[strike] = call_pain_sum + put_pain_sum

    if not total_pain_by_strike:
        return None, None, None, None, None

    max_pain_strike = min(total_pain_by_strike.items(), key=lambda x: x[1])[0]
    call_max_pain_strike = min(call_pain_by_strike.items(), key=lambda x: x[1])[0]
    put_max_pain_strike = min(put_pain_by_strike.items(), key=lambda x: x[1])[0]
    
    return (max_pain_strike, call_max_pain_strike, put_max_pain_strike, 
            total_pain_by_strike, call_pain_by_strike, put_pain_by_strike)

def create_max_pain_chart(calls, puts, S):
    """Create a chart showing max pain analysis with separate call and put pain."""
    result = calculate_max_pain(calls, puts)
    if result is None:
        return None
    
    (max_pain_strike, call_max_pain_strike, put_max_pain_strike,
     total_pain_by_strike, call_pain_by_strike, put_pain_by_strike) = result

    # Get colors from session state
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    padding = strike_range * 0.1
    
    fig = go.Figure()

    # Add total pain line (always vertical for max pain chart)
    # Use a neutral color that works well with both call and put colors
    total_pain_color = '#FFD700'  # Gold color for total pain
    
    if st.session_state.chart_type in ['Bar', 'Horizontal Bar']:
        fig.add_trace(go.Bar(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            name='Total Pain',
            marker_color=total_pain_color
        ))
    elif st.session_state.chart_type == 'Line':
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            mode='lines',
            name='Total Pain',
            line=dict(color=total_pain_color, width=2)
        ))
    elif st.session_state.chart_type == 'Scatter':
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            mode='markers',
            name='Total Pain',
            marker=dict(color=total_pain_color)
        ))
    else:  # Area
        fig.add_trace(go.Scatter(
            x=list(total_pain_by_strike.keys()),
            y=list(total_pain_by_strike.values()),
            fill='tozeroy',
            name='Total Pain',
            line=dict(color=total_pain_color, width=0.5),
            fillcolor='rgba(255, 215, 0, 0.3)'  # Semi-transparent gold
        ))

    # Add call pain line
    if st.session_state.show_calls:
        fig.add_trace(go.Scatter(
            x=list(call_pain_by_strike.keys()),
            y=list(call_pain_by_strike.values()),
            name='Call Pain',
            line=dict(color=call_color, width=1, dash='dot')
        ))

    # Add put pain line
    if st.session_state.show_puts:
        fig.add_trace(go.Scatter(
            x=list(put_pain_by_strike.keys()),
            y=list(put_pain_by_strike.values()),
            name='Put Pain',
            line=dict(color=put_color, width=1, dash='dot')
        ))

    # Calculate y-axis range with improved padding
    y_values = []
    for trace in fig.data:
        if hasattr(trace, 'y') and trace.y is not None:
            y_values.extend([y for y in trace.y if y is not None and not np.isnan(y)])
    
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        y_range = y_max - y_min
        
        # Add 15% padding on top and 5% on bottom (unless y_min is 0)
        padding_top = y_range * 0.15
        padding_bottom = y_range * 0.05 if y_min > 0 else 0
        y_min = y_min - padding_bottom
        y_max = y_max + padding_top
    else:
        # Default values if no valid y values
        y_min = 0
        y_max = 100

    # Add vertical lines for different max pain points
    fig.add_vline(
        x=max_pain_strike,
        line_dash="dash",
        line_color=total_pain_color,
        opacity=0.7,
        annotation_text=f"Total Max Pain: {max_pain_strike}",
        annotation_position="top"
    )

    if st.session_state.show_calls:
        fig.add_vline(
            x=call_max_pain_strike,
            line_dash="dash",
            line_color=call_color,
            opacity=0.7,
            annotation_text=f"Call Max Pain: {call_max_pain_strike}",
            annotation_position="top"
        )

    if st.session_state.show_puts:
        fig.add_vline(
            x=put_max_pain_strike,
            line_dash="dash",
            line_color=put_color,
            opacity=0.7,
            annotation_text=f"Put Max Pain: {put_max_pain_strike}",
            annotation_position="top"
        )

    # Add current price line
    fig.add_vline(
        x=S,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
        annotation_text=f"{S}",
        annotation_position="bottom"
    )

    fig.update_layout(
        title=dict(
            text='Max Pain',
            font=dict(size=st.session_state.chart_text_size + 8)  # Title slightly larger
        ),
        xaxis_title=dict(
            text='Strike Price',
            font=dict(size=st.session_state.chart_text_size)
        ),
        yaxis_title=dict(
            text='Total Pain',
            font=dict(size=st.session_state.chart_text_size)
        ),
        legend=dict(
            font=dict(size=st.session_state.chart_text_size)
        ),
        hovermode='x unified',
        xaxis=dict(
            autorange=True,
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        yaxis=dict(
            autorange=True,
            tickfont=dict(size=st.session_state.chart_text_size)
        ),
        height=600  # Increased height for better visibility
    )
    
    # Remove range slider for max pain chart as requested
    fig.update_xaxes(rangeslider=dict(visible=False))
    
    return fig

@st.cache_data(ttl=get_cache_ttl())  # Cache TTL matches refresh rate
def get_nearest_expiry(available_dates):
    """Get the nearest expiry date from a list of available dates"""
    if not available_dates:
        return None
    
    today = datetime.now().date()
    future_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in available_dates if datetime.strptime(date, '%Y-%m-%d').date() >= today]
    
    if not future_dates:
        return None
    
    return min(future_dates).strftime('%Y-%m-%d')

def create_davi_chart(calls, puts, S):
    """Create Delta-Adjusted Value Index chart that matches other exposure charts style"""
    # Get colors from session state
    call_color = st.session_state.call_color
    put_color = st.session_state.put_color

    # Create deep copies to avoid modifying original dataframes
    calls_df = calls.copy()
    puts_df = puts.copy()
    
    # Check if calc_delta column exists, if not, calculate delta
    if 'calc_delta' not in calls_df.columns:
        # Get current date and calculate time to expiration
        today = datetime.today().date()
        
        # Extract expiry date - use the first one if multiple
        if 'extracted_expiry' in calls_df.columns and not calls_df['extracted_expiry'].empty:
            selected_expiry = calls_df['extracted_expiry'].iloc[0]
            t_days = max((selected_expiry - today).days, 1)  # Ensure at least 1 day
            t = t_days / 365.0
            
            # Define function to compute delta
            def compute_delta(row, flag):
                sigma = row.get("impliedVolatility", None)
                if sigma is None or sigma <= 0:
                    return 0.5  # Default delta if IV is missing or invalid
                try:
                    delta_val, _, _ = calculate_greeks(flag, S, row["strike"], t, sigma)
                    return delta_val
                except Exception:
                    return 0.5  # Default delta if calculation fails
            
            # Calculate delta for calls and puts
            calls_df["calc_delta"] = calls_df.apply(lambda row: compute_delta(row, "c"), axis=1)
            puts_df["calc_delta"] = puts_df.apply(lambda row: compute_delta(row, "p"), axis=1)
        else:
            # If no expiry information, use approximate delta based on strike
            calls_df["calc_delta"] = calls_df.apply(lambda row: max(0, min(1, 1 - (row["strike"] - S) / (S * 0.1))), axis=1)
            puts_df["calc_delta"] = puts_df.apply(lambda row: max(0, min(1, (row["strike"] - S) / (S * 0.1))), axis=1)
    
    # Calculate DAVI for calls and puts with filtering
    # Only keep non-zero values
    calls_df['DAVI'] = (calls_df['volume'] + calls_df['openInterest']) * calls_df['lastPrice'] * calls_df['calc_delta']
    calls_df = calls_df[calls_df['DAVI'] != 0][['strike', 'DAVI']].copy()
    calls_df['OptionType'] = 'Call'

    puts_df['DAVI'] = (puts_df['volume'] + puts_df['openInterest']) * puts_df['lastPrice'] * puts_df['calc_delta']
    puts_df = puts_df[puts_df['DAVI'] != 0][['strike', 'DAVI']].copy()
    puts_df['OptionType'] = 'Put'

    # Calculate strike range around current price (percentage-based)
    strike_range = calculate_strike_range(S)
    min_strike = S - strike_range
    max_strike = S + strike_range
    
    # Filter data based on strike range
    calls_df = calls_df[(calls_df['strike'] >= min_strike) & (calls_df['strike'] <= max_strike)]
    puts_df = puts_df[(puts_df['strike'] >= min_strike) & (puts_df['strike'] <= max_strike)]

    # Calculate Net DAVI
    net_davi = pd.Series(0, index=sorted(set(calls_df['strike']) | set(puts_df['strike'])))
    if not calls_df.empty:
        net_davi = net_davi.add(calls_df.groupby('strike')['DAVI'].sum(), fill_value=0)
    if not puts_df.empty:
        net_davi = net_davi.add(puts_df.groupby('strike')['DAVI'].sum(), fill_value=0)

    # Calculate totals for title
    total_call_davi = calls_df['DAVI'].sum()
    total_put_davi = puts_df['DAVI'].sum()

    # Create title with totals
    title_with_totals = (
        f"Delta-Adjusted Value Index by Strike     "
        f"<span style='color: {call_color}'>{total_call_davi:,.0f}</span> | "
        f"<span style='color: {put_color}'>{total_put_davi:,.0f}</span>"
    )

    fig = go.Figure()

    # Add calls if enabled
    if st.session_state.show_calls:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                name='Call',
                marker_color=call_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=calls_df['strike'],
                x=calls_df['DAVI'],
                name='Call',
                marker_color=call_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                mode='markers',
                name='Call',
                marker=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                mode='lines',
                name='Call',
                line=dict(color=call_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=calls_df['strike'],
                y=calls_df['DAVI'],
                fill='tozeroy',
                name='Call',
                line=dict(color=call_color, width=0.5),
                fillcolor=call_color
            ))

    # Add puts if enabled
    if st.session_state.show_puts:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                name='Put',
                marker_color=put_color
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=puts_df['strike'],
                x=puts_df['DAVI'],
                name='Put',
                marker_color=put_color,
                orientation='h'
            ))
        elif st.session_state.chart_type == 'Scatter':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                mode='markers',
                name='Put',
                marker=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                mode='lines',
                name='Put',
                line=dict(color=put_color)
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=puts_df['strike'],
                y=puts_df['DAVI'],
                fill='tozeroy',
                name='Put',
                line=dict(color=put_color, width=0.5),
                fillcolor=put_color
            ))

    # Add Net if enabled
    if st.session_state.show_net and not net_davi.empty:
        if st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=net_davi.index,
                y=net_davi.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_davi.values]
            ))
        elif st.session_state.chart_type == 'Horizontal Bar':
            fig.add_trace(go.Bar(
                y=net_davi.index,
                x=net_davi.values,
                name='Net',
                marker_color=[call_color if val >= 0 else put_color for val in net_davi.values],
                orientation='h'
            ))
        elif st.session_state.chart_type in ['Scatter', 'Line']:
            positive_mask = net_davi.values >= 0
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[positive_mask],
                    y=net_davi.values[positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Positive)',
                    marker=dict(color=call_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=call_color) if st.session_state.chart_type == 'Line' else None
                ))
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[~positive_mask],
                    y=net_davi.values[~positive_mask],
                    mode='markers' if st.session_state.chart_type == 'Scatter' else 'lines',
                    name='Net (Negative)',
                    marker=dict(color=put_color) if st.session_state.chart_type == 'Scatter' else None,
                    line=dict(color=put_color) if st.session_state.chart_type == 'Line' else None
                ))
        elif st.session_state.chart_type == 'Area':
            positive_mask = net_davi.values >= 0
            if any(positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[positive_mask],
                    y=net_davi.values[positive_mask],
                    fill='tozeroy',
                    name='Net (Positive)',
                    line=dict(color=call_color, width=0.5),
                    fillcolor=call_color
                ))
            if any(~positive_mask):
                fig.add_trace(go.Scatter(
                    x=net_davi.index[~positive_mask],
                    y=net_davi.values[~positive_mask],
                    fill='tozeroy',
                    name='Net (Negative)',
                    line=dict(color=put_color, width=0.5),
                    fillcolor=put_color
                ))

    # Add current price line
    fig = add_current_price_line(fig, S)

    # Update layout
    padding = strike_range * 0.1
    if st.session_state.chart_type == 'Horizontal Bar':
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='DAVI',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='y unified',
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increased height for better visibility
        )
    else:
        fig.update_layout(
            title=dict(
                text=title_with_totals,
                x=0,
                xanchor='left',
                font=dict(size=st.session_state.chart_text_size + 8)
            ),
            xaxis_title=dict(
                text='Strike Price',
                font=dict(size=st.session_state.chart_text_size)
            ),
            yaxis_title=dict(
                text='DAVI',
                font=dict(size=st.session_state.chart_text_size)
            ),
            legend=dict(
                font=dict(size=st.session_state.chart_text_size)
            ),
            barmode='relative',
            hovermode='x unified',
            xaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            yaxis=dict(
                autorange=True,
                tickfont=dict(size=st.session_state.chart_text_size)
            ),
            height=600  # Increased height for better visibility
        )

    return fig

# Add at the start of each page's container
if st.session_state.current_page:
    market_status = check_market_status()
    if market_status:
        st.warning(market_status)

if st.session_state.current_page == "OI & Volume":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="options_data_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_oi"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache and expiry selections if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
            
            # Clear expiry selection state for current page when ticker changes
            page_expiry_key = f"{st.session_state.current_page}_selected_dates"
            if page_expiry_key in st.session_state:
                st.session_state[page_expiry_key] = []
            
            # Also clear any expiry selector widgets
            selector_key = f"{st.session_state.current_page}_expiry_selector"
            if selector_key in st.session_state:
                st.session_state[selector_key] = []
            
            # Force rerun to refresh available expiry dates
            st.rerun()
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.info("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: fetch_options_for_date(t, d, S)
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                # New: Add tabs to organize content
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["OI & Volume Charts", "Options Flow Analysis", "Premium Analysis", "Data Tables", "Market Maker"])
                
                with tab1:
                    # Original OI and Volume charts
                    oi_fig, volume_fig = create_oi_volume_charts(all_calls, all_puts, S)
                    st.plotly_chart(oi_fig, use_container_width=True)
                    st.plotly_chart(volume_fig, use_container_width=True)
                
                with tab2:
                    # New: Options flow analysis and visualizations
                    flow_data = analyze_options_flow(all_calls, all_puts, S)
                    flow_fig, money_fig, premium_fig, otm_fig = create_option_flow_charts(flow_data)
                    
                    # Create two columns for flow metrics
                    flow_col1, flow_col2 = st.columns(2)
                    
                    with flow_col1:
                        # Show bought vs sold volume
                        st.plotly_chart(flow_fig, use_container_width=True)
                        
                        # Show OTM vs ITM volume
                        st.plotly_chart(money_fig, use_container_width=True)
                    
                    with flow_col2:
                        # Show premium distribution
                        st.plotly_chart(premium_fig, use_container_width=True)
                        
                        # Show OTM analysis
                        st.plotly_chart(otm_fig, use_container_width=True)
                    
                    # Summary metrics display
                    st.subheader("Options Flow Summary")
                    
                    # Create a styled DataFrame for summary stats
                    summary_data = {
                        'Metric': [
                            'Total Call Premium', 'Total Put Premium',
                            'Call Volume', 'Put Volume',
                            'OTM Calls Bought', 'OTM Calls Sold',
                            'OTM Puts Bought', 'OTM Puts Sold'
                        ],
                        'Value': [
                            f"${flow_data['total_premium']['calls']:,.0f}",
                            f"${flow_data['total_premium']['puts']:,.0f}",
                            f"{flow_data['calls']['bought']['volume'] + flow_data['calls']['sold']['volume']:,.0f}",
                            f"{flow_data['puts']['bought']['volume'] + flow_data['puts']['sold']['volume']:,.0f}",
                            f"{flow_data['otm_detail']['calls_bought']:,.0f}",
                            f"{flow_data['otm_detail']['calls_sold']:,.0f}",
                            f"{flow_data['otm_detail']['puts_bought']:,.0f}",
                            f"{flow_data['otm_detail']['puts_sold']:,.0f}"
                        ],
                        'Type': [
                            'Call', 'Put',
                            'Call', 'Put',
                            'Call', 'Call',
                            'Put', 'Put'
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Apply styling with conditional formatting
                    def color_type(val):
                        color = st.session_state.call_color if val == 'Call' else st.session_state.put_color
                        return f'color: {color}'

                    # Display the table with styling
                    st.dataframe(
                        summary_df.style.map(lambda val: f'color: {st.session_state.call_color if val == "Call" else st.session_state.put_color}', subset=['Type']),
                        hide_index=True,
                        use_container_width=True
                    )
                    # Calculate and display put/call ratio
                    put_call_ratio = flow_data['puts']['bought']['volume'] / max(flow_data['calls']['bought']['volume'], 1)
                    
                    # Format the ratio as a string with 2 decimal places
                    ratio_str = f"{put_call_ratio:.2f}"
                    
                    # Determine background color based on ratio value
                    if put_call_ratio > 1.5:
                        bgcolor = "rgba(255, 0, 0, 0.2)"  # Red background for high put/call ratio
                    elif put_call_ratio < 0.7:
                        bgcolor = "rgba(0, 255, 0, 0.2)"  # Green background for low put/call ratio
                    else:
                        bgcolor = "rgba(255, 255, 255, 0.1)"  # Neutral background
                    
                    # Display put/call ratio in a styled container
                    st.markdown(
                        f"""
                        <div style="background-color: {bgcolor}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <h3 style="margin: 0;">Put/Call Volume Ratio: {ratio_str}</h3>
                            <p style="margin: 5px 0 0 0; font-size: 0.8em;">
                                {
                                    "Elevated put buying - possible bearish sentiment" if put_call_ratio > 1.5 else
                                    "More call buying than put buying - possible bullish sentiment" if put_call_ratio < 0.7 else
                                    "Neutral put/call ratio"
                                }
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with tab3:
                    # New: Advanced premium analysis
                    st.subheader("Premium Distribution Analysis")
                    
                    
                    # Premium summary statistics
                    total_call_premium = (all_calls['volume'] * all_calls['lastPrice'] * 100).sum()
                    total_put_premium = (all_puts['volume'] * all_puts['lastPrice'] * 100).sum()
                    premium_ratio = total_call_premium / max(total_put_premium, 1)  # Avoid division by zero
                    
                    # Premium by moneyness
                    all_calls['moneyness'] = all_calls.apply(lambda x: 'ITM' if x['strike'] <= S else 'OTM', axis=1)
                    all_puts['moneyness'] = all_puts.apply(lambda x: 'ITM' if x['strike'] >= S else 'OTM', axis=1)
                    
                    otm_call_premium = (all_calls[all_calls['moneyness'] == 'OTM']['volume'] * 
                                    all_calls[all_calls['moneyness'] == 'OTM']['lastPrice'] * 100).sum()
                    itm_call_premium = (all_calls[all_calls['moneyness'] == 'ITM']['volume'] * 
                                    all_calls[all_calls['moneyness'] == 'ITM']['lastPrice'] * 100).sum()
                    otm_put_premium = (all_puts[all_puts['moneyness'] == 'OTM']['volume'] * 
                                    all_puts[all_puts['moneyness'] == 'OTM']['lastPrice'] * 100).sum()
                    itm_put_premium = (all_puts[all_puts['moneyness'] == 'ITM']['volume'] * 
                                    all_puts[all_puts['moneyness'] == 'ITM']['lastPrice'] * 100).sum()
                    
                    # Calculate ITM premium flow
                    itm_net_premium = itm_call_premium - itm_put_premium
                    itm_premium_ratio = itm_call_premium / max(itm_put_premium, 1)
                    
                    # Determine ITM premium flow sentiment
                    if itm_premium_ratio > 1.5:
                        itm_sentiment = "Bullish"
                        itm_color = st.session_state.call_color
                    elif itm_premium_ratio < 0.7:
                        itm_sentiment = "Bearish"
                        itm_color = st.session_state.put_color
                    else:
                        itm_sentiment = "Neutral"
                        itm_color = "white"
                        
                    # Display premium metrics in a cleaner format with call/put ratio indicator
                    st.markdown("### Premium Summary")
                    
                    # Call/put premium ratio status indicator
                    ratio_status = ""
                    if premium_ratio > 1.5:
                        ratio_status = "Bullish (high call premium)"
                        ratio_color = st.session_state.call_color
                    elif premium_ratio < 0.7:
                        ratio_status = "Bearish (high put premium)"
                        ratio_color = st.session_state.put_color
                    else:
                        ratio_status = "Neutral"
                        ratio_color = "white"
                    
                    # Create metrics with custom styling
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: rgba(50,50,50,0.3); margin-bottom: 15px;">
                            <h4>Call/Put Premium Ratio: <span style="color: {ratio_color}">{premium_ratio:.2f}</span> {ratio_status}</h4>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Add ITM premium flow indicator
                    st.markdown(
                        f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: rgba(50,50,50,0.3); margin-bottom: 15px;">
                            <h4>ITM Premium Flow: <span style="color: {itm_color}">{itm_sentiment}</span> (Call/Put Ratio: {itm_premium_ratio:.2f})</h4>
                            <p>Net ITM Premium: <span style="color: {st.session_state.call_color if itm_net_premium > 0 else st.session_state.put_color}">${abs(itm_net_premium):,.0f}</span> 
                            {" toward calls" if itm_net_premium > 0 else " toward puts"}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Create three columns for better organization
                    premium_col1, premium_col2, premium_col3 = st.columns(3)
                    
                    with premium_col1:
                        st.markdown(f"<h4 style='color: {st.session_state.call_color}'>Call Premium</h4>", unsafe_allow_html=True)
                        st.metric("Total", f"${total_call_premium:,.0f}")
                        st.metric("OTM", f"${otm_call_premium:,.0f}", f"{otm_call_premium/total_call_premium*100:.1f}%" if total_call_premium > 0 else "0%")
                        st.metric("ITM", f"${itm_call_premium:,.0f}", f"{itm_call_premium/total_call_premium*100:.1f}%" if total_call_premium > 0 else "0%")
                    
                    with premium_col2:
                        st.markdown(f"<h4 style='color: {st.session_state.put_color}'>Put Premium</h4>", unsafe_allow_html=True)
                        st.metric("Total", f"${total_put_premium:,.0f}")
                        st.metric("OTM", f"${otm_put_premium:,.0f}", f"{otm_put_premium/total_put_premium*100:.1f}%" if total_put_premium > 0 else "0%")
                        st.metric("ITM", f"${itm_put_premium:,.0f}", f"{itm_put_premium/total_put_premium*100:.1f}%" if total_put_premium > 0 else "0%")
                    
                    with premium_col3:
                        st.markdown("<h4>Premium Analysis</h4>", unsafe_allow_html=True)
                        
                        # Calculate OTM/ITM ratios for sentiment analysis
                        otm_itm_call_ratio = otm_call_premium / max(itm_call_premium, 1)
                        otm_itm_put_ratio = otm_put_premium / max(itm_put_premium, 1)
                        
                        # Show OTM to ITM ratios
                        st.metric("OTM/ITM Call Ratio", f"{otm_itm_call_ratio:.2f}")
                        st.metric("OTM/ITM Put Ratio", f"{otm_itm_put_ratio:.2f}")
                        
                        # Premium concentration
                        total_premium = total_call_premium + total_put_premium
                        st.metric("Call Premium %", f"{total_call_premium/total_premium*100:.1f}%" if total_premium > 0 else "0%")
                        st.metric("Put Premium %", f"{total_put_premium/total_premium*100:.1f}%" if total_premium > 0 else "0%")
                        
                    # Add new ITM premium flow analysis chart
                    st.markdown("### ITM Premium Flow Analysis")
                    
                    # Create a bar chart to visualize ITM premium distribution
                    itm_premium_fig = go.Figure()
                    
                    # Add bars for ITM Call and Put premium
                    itm_premium_fig.add_trace(go.Bar(
                        x=['ITM Calls'],
                        y=[itm_call_premium],
                        name='ITM Call Premium',
                        marker_color=st.session_state.call_color
                    ))
                    
                    itm_premium_fig.add_trace(go.Bar(
                        x=['ITM Puts'],
                        y=[itm_put_premium],
                        name='ITM Put Premium',
                        marker_color=st.session_state.put_color
                    ))
                    
                    # Add a third bar for net ITM premium flow if needed
                    itm_premium_fig.add_trace(go.Bar(
                        x=['Net ITM Flow'],
                        y=[itm_net_premium],
                        name='Net ITM Premium',
                        marker_color=st.session_state.call_color if itm_net_premium > 0 else st.session_state.put_color
                    ))
                    
                    itm_premium_fig.update_layout(
                        title=dict(
                            text=f"ITM Premium Flow - {itm_sentiment}",
                            font=dict(size=st.session_state.chart_text_size + 6)
                        ),
                        xaxis_title=dict(
                            text="Option Type",
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        yaxis_title=dict(
                            text="Premium ($)",
                            font=dict(size=st.session_state.chart_text_size)
                        ),
                        template="plotly_dark",
                        xaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size)
                        ),
                        yaxis=dict(
                            tickfont=dict(size=st.session_state.chart_text_size)
                        )
                    )
                    
                    st.plotly_chart(itm_premium_fig, use_container_width=True)
                    
                    # Add additional premium insights with ITM flow details
                    st.markdown("### Premium Insights")
                    
                    # Create strike-based premium insights - Fix for deprecation warning
                    # Instead of using groupby().apply(), calculate premium directly
                    calls_premium = all_calls.copy()
                    calls_premium['premium'] = calls_premium['volume'] * calls_premium['lastPrice'] * 100
                    call_premium_by_strike = calls_premium.groupby('strike')['premium'].sum().reset_index()
                    
                    puts_premium = all_puts.copy()
                    puts_premium['premium'] = puts_premium['volume'] * puts_premium['lastPrice'] * 100
                    put_premium_by_strike = puts_premium.groupby('strike')['premium'].sum().reset_index()
                    
                    # Find top premium concentrations
                    top_call_strikes = call_premium_by_strike.nlargest(5, 'premium')
                    top_put_strikes = put_premium_by_strike.nlargest(5, 'premium')
                    
                    # Calculate call vs put premium for each strike and net premium flow
                    premium_combined = pd.merge(call_premium_by_strike, put_premium_by_strike, on='strike', how='outer', suffixes=('_call', '_put')).fillna(0)
                    premium_combined['net_premium'] = premium_combined['premium_call'] - premium_combined['premium_put']
                    premium_combined['ratio'] = premium_combined['premium_call'] / premium_combined['premium_put'].replace(0, 1)
                    
                    # Find strikes with most bullish and bearish premium flow
                    bullish_strikes = premium_combined.nlargest(5, 'net_premium')
                    bearish_strikes = premium_combined.nsmallest(5, 'net_premium')
                    
                    # Calculate total premium for percentages
                    total_call_premium_sum = call_premium_by_strike['premium'].sum()
                    total_put_premium_sum = put_premium_by_strike['premium'].sum()
                    
                    insight_col1, insight_col2 = st.columns(2)
                    
                    # Use a table format instead of text lines to avoid formatting issues
                    with insight_col1:
                        st.markdown(f"<h5 style='color: {st.session_state.call_color}'>Top Call Premium Strikes</h5>", unsafe_allow_html=True)
                        
                        # Create DataFrames for display
                        call_strikes_data = []
                        for _, row in top_call_strikes.iterrows():
                            pct = (row['premium'] / total_call_premium_sum * 100) if total_call_premium_sum > 0 else 0
                            call_strikes_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Premium": f"${row['premium']:,.0f}", 
                                "% of Total": f"{pct:.1f}%"
                            })
                        st.table(pd.DataFrame(call_strikes_data))
                        
                        st.markdown(f"<h5 style='color: {st.session_state.call_color}'>Most Bullish Premium Flow</h5>", unsafe_allow_html=True)
                        
                        bullish_data = []
                        for _, row in bullish_strikes.iterrows():
                            bullish_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Net Premium": f"+${row['net_premium']:,.0f}", 
                                "C/P Ratio": f"{row['ratio']:.2f}"
                            })
                        st.table(pd.DataFrame(bullish_data))
                    
                    with insight_col2:
                        st.markdown(f"<h5 style='color: {st.session_state.put_color}'>Top Put Premium Strikes</h5>", unsafe_allow_html=True)
                        
                        put_strikes_data = []
                        for _, row in top_put_strikes.iterrows():
                            pct = (row['premium'] / total_put_premium_sum * 100) if total_put_premium_sum > 0 else 0
                            put_strikes_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Premium": f"${row['premium']:,.0f}", 
                                "% of Total": f"{pct:.1f}%"
                            })
                        st.table(pd.DataFrame(put_strikes_data))
                        
                        st.markdown(f"<h5 style='color: {st.session_state.put_color}'>Most Bearish Premium Flow</h5>", unsafe_allow_html=True)
                        
                        bearish_data = []
                        for _, row in bearish_strikes.iterrows():
                            bearish_data.append({
                                "Strike": f"${row['strike']:.1f}", 
                                "Net Premium": f"-${abs(row['net_premium']):,.0f}", 
                                "C/P Ratio": f"{row['ratio']:.2f}"
                            })
                        st.table(pd.DataFrame(bearish_data))

                with tab4:
                    # Original data tables
                    volume_over_oi = st.checkbox("Show only rows where Volume > Open Interest")
                    # Filter data based on volume over OI if checked
                    calls_filtered = all_calls.copy()
                    puts_filtered = all_puts.copy()
                    if volume_over_oi:
                        calls_filtered = calls_filtered[calls_filtered['volume'] > calls_filtered['openInterest']]
                        puts_filtered = puts_filtered[puts_filtered['volume'] > puts_filtered['openInterest']]
                    if calls_filtered.empty and puts_filtered.empty:
                        st.warning("No data left after applying filters.")
                    else:
                        st.write("### Filtered Data Tables")
                        if not calls_filtered.empty:
                            st.write("**Calls Table**")
                            st.dataframe(calls_filtered)
                        else:
                            st.write("No calls match filters.")
                        if not puts_filtered.empty:
                            st.write("**Puts Table**")
                            st.dataframe(puts_filtered)
                        else:
                            st.write("No puts match filters.")
                
                with tab5:
                    # Market Maker tab content
                    st.write("### 📊 Market Maker Positioning")
                    st.write(f"**Symbol:** {ticker.upper()} | **Expiry:** {', '.join(selected_expiry_dates)}")
                    st.info("📅 **Data Source**: OCC (Options Clearing Corporation) | **Timing**: Latest business day from past 24 hours")
                    
                    # Add informational section about OCC and data timing
                    with st.expander("ℹ️ About OCC Market Maker Data", expanded=False):
                        st.markdown("""
                        **What is the OCC (Options Clearing Corporation)?**
                        
                        The OCC is the world's largest equity derivatives clearing organization and acts as the central counterparty for all options trades in the U.S. They guarantee the performance of all options contracts and provide transparency into market activity.
                        
                        **Market Maker Role:**
                        - Market makers provide liquidity by continuously quoting bid and ask prices
                        - They facilitate trading by being ready to buy or sell options contracts
                        - Their positioning data reveals institutional sentiment and flow direction
                        
                        **Data Timing & Availability:**
                        - 📅 **Report Date**: Data is from the latest business day within the past 24 hours
                        - ⏰ **Update Schedule**: OCC updates this data daily after market close
                        - 🕐 **Lag Time**: Data typically reflects previous trading day activity
                        - 📊 **Coverage**: All option types (equity, index, ETF options)
                        
                        **Why This Matters:**
                        - Market maker positioning can indicate institutional sentiment
                        - Large call positions may suggest bullish positioning
                        - Large put positions may suggest bearish positioning or hedging activity
                        - Combined with other analysis, helps understand market dynamics
                        
                        **Important Notes:**
                        - This is historical data (not real-time)
                        - Market maker positions can change rapidly during trading hours
                        - Data should be used in conjunction with other analysis tools
                        """)
                    
                    st.write("")  # Add spacing
                    
                    # Initialize session state for market maker data
                    if 'mm_data' not in st.session_state:
                        st.session_state.mm_data = None
                        st.session_state.mm_message = None
                        st.session_state.mm_last_ticker = None
                        st.session_state.mm_last_expiry = None
                    
                    # Check if we need to fetch new data (ticker or expiry changed)
                    current_expiries = tuple(selected_expiry_dates) if selected_expiry_dates else None
                    need_refresh = (
                        st.session_state.mm_last_ticker != ticker or 
                        st.session_state.mm_last_expiry != current_expiries or
                        st.session_state.mm_data is None
                    )
                    
                    # Auto-fetch data when symbol or expiry changes
                    if need_refresh and ticker and current_expiries:
                        with st.spinner("Loading market maker data for multiple expiration dates..."):
                            combined_data = []
                            success_messages = []
                            
                            # Fetch data for each selected expiration date
                            for expiry_date in selected_expiry_dates:
                                data, message = download_volume_csv(ticker, "U", expiry_date)
                                if data:
                                    combined_data.append(data)
                                    success_messages.append(f"✓ {expiry_date}")
                                else:
                                    success_messages.append(f"✗ {expiry_date}: {message}")
                            
                            # Combine all CSV data
                            if combined_data:
                                # Join all CSV data with headers
                                all_csv_data = []
                                for i, csv_data in enumerate(combined_data):
                                    lines = csv_data.strip().split('\n')
                                    # Skip header lines and add only data lines
                                    data_lines = []
                                    for line in lines:
                                        if ',' in line and line.strip() and line.strip()[0].isdigit():
                                            data_lines.extend(lines[lines.index(line):])
                                            break
                                    all_csv_data.extend(data_lines)
                                
                                combined_csv = '\n'.join(all_csv_data)
                                combined_message = f"Market Maker Data Retrieved for {len(selected_expiry_dates)} expiration dates:\n" + '\n'.join(success_messages)
                                
                                st.session_state.mm_data = combined_csv
                                st.session_state.mm_message = combined_message
                            else:
                                st.session_state.mm_data = None
                                st.session_state.mm_message = "Failed to retrieve data for any selected expiration dates."
                            
                            st.session_state.mm_last_ticker = ticker
                            st.session_state.mm_last_expiry = current_expiries
                    

                    # Display results
                    if st.session_state.mm_message:
                        if st.session_state.mm_data:
                            st.success(st.session_state.mm_message)
                            
                            # Process the market maker data
                            summary_data = process_market_maker_data(st.session_state.mm_data)
                            
                            if summary_data:
                                # Display summary metrics
                                st.write("### 📊 Market Maker Summary")
                                
                                # Create metric columns
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric(
                                        label="Total Volume",
                                        value=f"{summary_data['total_volume']:,}"
                                    )
                                
                                with col2:
                                    call_color = st.session_state.call_color
                                    st.markdown(f"""
                                    <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;">
                                        <p style="margin: 0; font-size: 14px; color: #888;">Call Volume</p>
                                        <p style="margin: 5px 0; font-size: 24px; font-weight: bold; color: {call_color};">{summary_data['call_volume']:,}</p>
                                        <p style="margin: 0; font-size: 14px; color: {call_color};">{summary_data['call_percentage']:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    put_color = st.session_state.put_color
                                    st.markdown(f"""
                                    <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;">
                                        <p style="margin: 0; font-size: 14px; color: #888;">Put Volume</p>
                                        <p style="margin: 5px 0; font-size: 24px; font-weight: bold; color: {put_color};">{summary_data['put_volume']:,}</p>
                                        <p style="margin: 0; font-size: 14px; color: {put_color};">{summary_data['put_percentage']:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col4:
                                    if summary_data['call_volume'] > summary_data['put_volume']:
                                        bias = "Call Bias"
                                        bias_pct = summary_data['call_percentage'] - 50
                                        bias_color = "#00FF00"  # Green for call bias
                                    else:
                                        bias = "Put Bias"
                                        bias_pct = summary_data['put_percentage'] - 50
                                        bias_color = "#FF0000"  # Red for put bias
                                    
                                    st.markdown(f"""
                                    <div style="border: 1px solid #333; border-radius: 5px; padding: 10px; text-align: center;">
                                        <p style="margin: 0; font-size: 14px; color: #888;">Market Bias</p>
                                        <p style="margin: 5px 0; font-size: 24px; font-weight: bold; color: {bias_color};">{bias}</p>
                                        <p style="margin: 0; font-size: 14px; color: {bias_color};">{'+' if bias_pct > 0 else ''}{bias_pct:.1f}%</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                st.write("---")
                                
                                # Create and display charts
                                if summary_data['total_volume'] > 0:
                                    st.write("### 📈 Visual Analysis")
                                    
                                    fig_pie, fig_bar = create_market_maker_charts(summary_data)
                                    
                                    # Display charts in columns
                                    chart_col1, chart_col2 = st.columns(2)
                                    
                                    with chart_col1:
                                        st.plotly_chart(fig_pie, use_container_width=True)
                                    
                                    with chart_col2:
                                        st.plotly_chart(fig_bar, use_container_width=True)
                                
                                # Optional: Show data table in collapsible section
                                with st.expander("📋 View Raw Data Table", expanded=False):
                                    if not summary_data['raw_data'].empty:
                                        st.dataframe(summary_data['raw_data'], use_container_width=True)
                                    else:
                                        st.warning("No data available in the table.")
                            else:
                                st.warning("Unable to process market maker data. The data format may not be recognized.")
                                
                        else:
                            st.error(st.session_state.mm_message)
                    else:
                        # Show loading state or instructions
                        if not ticker:
                            st.info("💡 Enter a symbol above to view market maker positioning data.")
                        elif not current_expiries:
                            st.info("💡 Select expiration date(s) to view market maker positioning data.")
                        else:
                            st.info("🔄 Loading market maker data automatically...")
                        
                        # Show some help information
                        with st.expander("ℹ️ About Market Maker Data"):
                            st.write("""
                            **Market Maker Positioning Data from OCC:**
                            
                            - **Source**: Options Clearing Corporation (OCC)
                            - **Data Type**: Market maker volume and positioning
                            - **Update Frequency**: Daily (business days only)
                            - **Report Date**: Latest business day from 24 hours ago
                            - **Coverage**: All option types (Equity, Index, etc.)
                            
                            **How it works:**
                            - Data loads automatically when you enter a symbol and select expiration date(s)
                            - Supports multiple expiration dates - data is combined automatically
                            - Data refreshes automatically when you change the symbol or expiration selection
                            
                            **Data shows:**
                            - Call vs Put volume distribution
                            - Market maker activity breakdown
                            - Symbol-specific positioning data
                            
                            **Note**: Data availability depends on market maker activity and OCC reporting schedules.
                            """)

elif st.session_state.current_page == "Gamma Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Vanna Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Delta Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Charm Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Speed Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Vomma Exposure":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = st.session_state.current_page.split()[0].lower()  # gamma, vanna, delta, charm, speed, or vomma
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key=f"{page_name}_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key=f"refresh_button_{page_name}"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                exposure_type_map = {
                    "Gamma Exposure": "GEX",
                    "Vanna Exposure": "VEX",
                    "Delta Exposure": "DEX",
                    "Charm Exposure": "Charm",
                    "Speed Exposure": "Speed",
                    "Vomma Exposure": "Vomma"
                }
                
                exposure_type = exposure_type_map[st.session_state.current_page]
                
                # Modify the bar chart title to show multiple dates
                title = f"{st.session_state.current_page} by Strike ({len(selected_expiry_dates)} dates)"
                fig_bar = create_exposure_bar_chart(all_calls, all_puts, exposure_type, title, S)
                st.plotly_chart(fig_bar, use_container_width=True)

elif st.session_state.current_page == "Exposure by Notional Value":
    exposure_container = st.container()
    with exposure_container:
        st.empty()  # Clear previous content
        page_name = "notional"  # Use consistent page name for compute_greeks_and_charts
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="notional_exposure_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_notional"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, page_name, S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                # Calculate notional value exposure from raw greek values
                def calculate_notional_exposure(df, exposure_col):
                    """Calculate notional value exposure from raw greek values × contract price × contract size"""
                    if exposure_col not in df.columns:
                        return df
                    
                    # Use lastPrice if available, otherwise use ask price
                    price_col = 'lastPrice' if 'lastPrice' in df.columns else 'ask'
                    
                    # Determine which volume metric to use
                    volume_metric = 'volume' if st.session_state.get('use_volume_for_greeks', False) else 'openInterest'
                    
                    # Calculate notional exposure from raw greek values (not the already-scaled exposure values)
                    # This avoids double-scaling issues
                    if exposure_col == "GEX":
                        # Notional = Gamma × Volume/OI × Contract Size × Contract Price
                        df[f'{exposure_col}_notional'] = df['calc_gamma'] * df[volume_metric] * 100 * df[price_col]
                    elif exposure_col == "VEX":
                        # Notional = Vanna × Volume/OI × Contract Size × Contract Price
                        df[f'{exposure_col}_notional'] = df['calc_vanna'] * df[volume_metric] * 100 * df[price_col]
                    elif exposure_col == "DEX":
                        # Notional = Delta × Volume/OI × Contract Size × Contract Price
                        df[f'{exposure_col}_notional'] = df['calc_delta'] * df[volume_metric] * 100 * df[price_col]
                    elif exposure_col == "Charm":
                        # Notional = Charm × Volume/OI × Contract Size × Contract Price
                        df[f'{exposure_col}_notional'] = df['calc_charm'] * df[volume_metric] * 100 * df[price_col]
                    elif exposure_col == "Speed":
                        # Notional = Speed × Volume/OI × Contract Size × Contract Price
                        df[f'{exposure_col}_notional'] = df['calc_speed'] * df[volume_metric] * 100 * df[price_col]
                    elif exposure_col == "Vomma":
                        # Notional = Vomma × Volume/OI × Contract Size × Contract Price
                        df[f'{exposure_col}_notional'] = df['calc_vomma'] * df[volume_metric] * 100 * df[price_col]
                    
                    return df
                
                # Calculate notional exposure for all exposure types
                for exposure_type in ["GEX", "VEX", "DEX", "Charm", "Speed", "Vomma"]:
                    if f'calc_{exposure_type.lower()}' in all_calls.columns or exposure_type in all_calls.columns:
                        all_calls = calculate_notional_exposure(all_calls, exposure_type)
                        all_puts = calculate_notional_exposure(all_puts, exposure_type)
                
                # Create tabs for different exposure types
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Gamma (GEX)", "Vanna (VEX)", "Delta (DEX)", "Charm", "Speed", "Vomma"])
                
                with tab1:
                    if "GEX_notional" in all_calls.columns:
                        title = f"GEX Notional Value Exposure by Strike ({len(selected_expiry_dates)} dates)"
                        fig_gex = create_exposure_bar_chart(all_calls, all_puts, "GEX_notional", title, S)
                        st.plotly_chart(fig_gex, use_container_width=True)
                    else:
                        st.warning("GEX data not available.")
                
                with tab2:
                    if "VEX_notional" in all_calls.columns:
                        title = f"VEX Notional Value Exposure by Strike ({len(selected_expiry_dates)} dates)"
                        fig_vex = create_exposure_bar_chart(all_calls, all_puts, "VEX_notional", title, S)
                        st.plotly_chart(fig_vex, use_container_width=True)
                    else:
                        st.warning("VEX data not available.")
                
                with tab3:
                    if "DEX_notional" in all_calls.columns:
                        title = f"DEX Notional Value Exposure by Strike ({len(selected_expiry_dates)} dates)"
                        fig_dex = create_exposure_bar_chart(all_calls, all_puts, "DEX_notional", title, S)
                        st.plotly_chart(fig_dex, use_container_width=True)
                    else:
                        st.warning("DEX data not available.")
                
                with tab4:
                    if "Charm_notional" in all_calls.columns:
                        title = f"Charm Notional Value Exposure by Strike ({len(selected_expiry_dates)} dates)"
                        fig_charm = create_exposure_bar_chart(all_calls, all_puts, "Charm_notional", title, S)
                        st.plotly_chart(fig_charm, use_container_width=True)
                    else:
                        st.warning("Charm data not available.")
                
                with tab5:
                    if "Speed_notional" in all_calls.columns:
                        title = f"Speed Notional Value Exposure by Strike ({len(selected_expiry_dates)} dates)"
                        fig_speed = create_exposure_bar_chart(all_calls, all_puts, "Speed_notional", title, S)
                        st.plotly_chart(fig_speed, use_container_width=True)
                    else:
                        st.warning("Speed data not available.")
                
                with tab6:
                    if "Vomma_notional" in all_calls.columns:
                        title = f"Vomma Notional Value Exposure by Strike ({len(selected_expiry_dates)} dates)"
                        fig_vomma = create_exposure_bar_chart(all_calls, all_puts, "Vomma_notional", title, S)
                        st.plotly_chart(fig_vomma, use_container_width=True)
                    else:
                        st.warning("Vomma data not available.")

elif st.session_state.current_page == "Calculated Greeks":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        st.write("This page calculates delta, gamma, and vanna based on market data.")
        
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="calculated_greeks_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_greeks"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                # Get nearest expiry and use it as default
                nearest_expiry = get_nearest_expiry(available_dates)
                expiry_date_str = st.selectbox(
                    "Select an Exp. Date:", 
                    options=available_dates, 
                    index=available_dates.index(nearest_expiry) if nearest_expiry else None, 
                    key="calculated_greeks_expiry_main"
                )
                
                if expiry_date_str:  # Only proceed if an expiry date is selected
                    
                    selected_expiry = datetime.strptime(expiry_date_str, '%Y-%m-%d').date()
                    calls, puts = fetch_options_for_date(ticker, expiry_date_str, S)
                    
                    if calls.empty and puts.empty:
                        st.warning("No options data available for this ticker.")
                        st.stop()

                    # Rest of the Calculated Greeks logic
                    combined = pd.concat([calls, puts])
                    combined = combined.dropna(subset=['extracted_expiry'])
                    calls = calls[calls['extracted_expiry'] == selected_expiry]
                    puts = puts[puts['extracted_expiry'] == selected_expiry]
                    
                    # Get stock price
                    stock = yf.Ticker(ticker)
                    S = get_last_price(stock)
                    if S is None:
                        st.error("Could not fetch underlying price.")
                        st.stop()

                    S = round(S, 2)
                    st.markdown(f"**Underlying Price (S):** {S}")
                    
                    today = datetime.today().date()
                    t_days = (selected_expiry - today).days
                    # Change this condition to allow same-day expiration
                    if t_days < 0:  # Changed from t_days <= 0
                        st.error("The selected expiration date is in the past!")
                        st.stop()

                    t = t_days / 365.0
                    st.markdown(f"**Time to Expiration (t in years):** {t:.4f}")
                    
                    def compute_row_greeks(row, flag):
                        try:
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return pd.Series({"calc_delta": None, "calc_gamma": None, "calc_vanna": None})
                            
                            delta_val, gamma_val, vanna_val = calculate_greeks(flag, S, row["strike"], t, sigma)
                            return pd.Series({
                                "calc_delta": delta_val,
                                "calc_gamma": gamma_val,
                                "calc_vanna": vanna_val
                            })
                        except Exception as e:
                            st.warning(f"Error calculating greeks: {str(e)}")
                            return pd.Series({"calc_delta": None, "calc_gamma": None, "calc_vanna": None})

                    results = {}
                    
                    # Process calls
                    if not calls.empty:
                        try:
                            calls_copy = calls.copy()
                            greeks_calls = calls_copy.apply(lambda row: compute_row_greeks(row, "c"), axis=1)
                            results["Calls"] = pd.concat([calls_copy, greeks_calls], axis=1)
                        except Exception as e:
                            st.warning(f"Error processing calls: {str(e)}")
                    else:
                        st.warning("No call options data available.")

                    # Process puts
                    if not puts.empty:
                        try:
                            puts_copy = puts.copy()
                            greeks_puts = puts_copy.apply(lambda row: compute_row_greeks(row, "p"), axis=1)
                            results["Puts"] = pd.concat([puts_copy, greeks_puts], axis=1)
                        except Exception as e:
                            st.warning(f"Error processing puts: {str(e)}")
                    else:
                        st.warning("No put options data available.")

                    # Display results
                    for typ, df in results.items():
                        try:
                            st.write(f"### {typ} with Calculated Greeks")
                            st.dataframe(df[['contractSymbol', 'strike', 'impliedVolatility', 'calc_delta', 'calc_gamma', 'calc_vanna']])
                            fig = px.scatter(df, x="strike", y="calc_delta", title=f"{typ}: Delta vs. Strike",
                                         labels={"strike": "Strike", "calc_delta": "Calculated Delta"})
                            st.plotly_chart(fig, use_container_width=True, key=f"Calculated Greeks_{typ.lower()}_scatter")
                        except Exception as e:
                            st.error(f"Error displaying {typ} data: {str(e)}")
                else:
                    st.warning("Please select an expiration date to view the calculations.")
                    st.stop()

if st.session_state.current_page == "Dashboard":
    dashboard_container = st.container()
    with dashboard_container:
        st.empty()  # Clear previous content
        
        # Create a single input for ticker with refresh button
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="dashboard_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_dashboard"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                # Get nearest expiry and use it as default
                nearest_expiry = get_nearest_expiry(available_dates)
                expiry_date_str = st.selectbox(
                    "Select an Exp. Date:", 
                    options=available_dates,
                    index=available_dates.index(nearest_expiry) if nearest_expiry else None,
                    key="dashboard_expiry_main"
                )
                
                if expiry_date_str:  # Only proceed if expiry date is selected
                    calls, puts, _, t, selected_expiry, today = compute_greeks_and_charts(ticker, expiry_date_str, "dashboard", S)
                    if calls is None or puts is None:
                        st.stop()
                        
                    fig_gamma = create_exposure_bar_chart(calls, puts, "GEX", "Gamma Exposure by Strike", S)
                    fig_vanna = create_exposure_bar_chart(calls, puts, "VEX", "Vanna Exposure by Strike", S)
                    fig_delta = create_exposure_bar_chart(calls, puts, "DEX", "Delta Exposure by Strike", S)
                    fig_charm = create_exposure_bar_chart(calls, puts, "Charm", "Charm Exposure by Strike", S)
                    fig_speed = create_exposure_bar_chart(calls, puts, "Speed", "Speed Exposure by Strike", S)
                    fig_vomma = create_exposure_bar_chart(calls, puts, "Vomma", "Vomma Exposure by Strike", S)
                    
                    # Intraday price chart
                    intraday_data, current_price, vix_data = get_combined_intraday_data(ticker)
                    if intraday_data is None or current_price is None:
                        st.warning("No intraday data available for this ticker.")
                    else:
                        # Initialize plot with cleared shapes/annotations
                        fig_intraday = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_intraday.layout.shapes = []
                        fig_intraday.layout.annotations = []

                        # Add either candlestick or line trace based on selection
                        if st.session_state.intraday_chart_type == 'Candlestick':
                            if st.session_state.candlestick_type == 'Heikin Ashi':
                                # Calculate Heikin Ashi values
                                ha_data = calculate_heikin_ashi(intraday_data)
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=ha_data.index,
                                        open=ha_data['HA_Open'],
                                        high=ha_data['HA_High'],
                                        low=ha_data['HA_Low'],
                                        close=ha_data['HA_Close'],
                                        name="Price",
                                        increasing_line_color=st.session_state.call_color,
                                        decreasing_line_color=st.session_state.put_color,
                                        increasing_fillcolor=st.session_state.call_color,
                                        decreasing_fillcolor=st.session_state.put_color,
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )
                            elif st.session_state.candlestick_type == 'Hollow':
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=intraday_data.index,
                                        open=intraday_data['Open'],
                                        high=intraday_data['High'],
                                        low=intraday_data['Low'],
                                        close=intraday_data['Close'],
                                        name="Price",
                                        increasing=dict(line=dict(color=st.session_state.call_color), fillcolor='rgba(0,0,0,0)'),
                                        decreasing=dict(line=dict(color=st.session_state.put_color), fillcolor='rgba(0,0,0,0)'),
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )
                            else:  # Filled candlesticks
                                fig_intraday.add_trace(
                                    go.Candlestick(
                                        x=intraday_data.index,
                                        open=intraday_data['Open'],
                                        high=intraday_data['High'],
                                        low=intraday_data['Low'],
                                        close=intraday_data['Close'],
                                        name="Price",
                                        increasing_line_color=st.session_state.call_color,
                                        decreasing_line_color=st.session_state.put_color,
                                        increasing_fillcolor=st.session_state.call_color,
                                        decreasing_fillcolor=st.session_state.put_color,
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )
                        else:  # Line chart
                            fig_intraday.add_trace(
                                go.Scatter(
                                    x=intraday_data.index,
                                    y=intraday_data['Close'],
                                    name="Price",
                                    line=dict(color='gold'),
                                    showlegend=False
                                ),
                                secondary_y=False
                            )

                        # Add technical indicators if enabled
                        if st.session_state.get('show_technical_indicators') and st.session_state.get('selected_indicators'):
                            # Calculate technical indicators
                            indicators = calculate_technical_indicators(intraday_data)
                            
                            # Calculate Fibonacci levels if selected
                            fibonacci_levels = None
                            if "Fibonacci Retracements" in st.session_state.selected_indicators:
                                fibonacci_levels = calculate_fibonacci_levels(intraday_data)
                            
                            # Add indicators to chart
                            fig_intraday = add_technical_indicators_to_chart(fig_intraday, indicators, fibonacci_levels)

                        # Calculate base y-axis range from price data
                        price_min = intraday_data['Low'].min()
                        price_max = intraday_data['High'].max()
                        price_range = price_max - price_min
                        padding = price_range * 0.1  # 10% padding
                        y_min = price_min - padding
                        y_max = price_max + padding

                        # Add VIX overlay if enabled
                        if st.session_state.show_vix_overlay and vix_data is not None and not vix_data.empty:
                            vix_min = vix_data['Close'].min()
                            vix_max = vix_data['Close'].max()
                            vix_range = vix_max - vix_min

                            if vix_range == 0:
                                st.warning("VIX data has no range, overlay disabled")
                            else:
                                # Normalize VIX to fit within 50% of price range, centered
                                target_vix_range = price_range * 0.5
                                vix_midpoint = (price_max + price_min) / 2
                                normalized_vix = vix_midpoint + (((vix_data['Close'] - vix_min) / vix_range - 0.5) * target_vix_range)


                                fig_intraday.add_trace(
                                    go.Scatter(
                                        x=vix_data.index,
                                        y=normalized_vix,
                                        name='VIX',
                                        line=dict(color=st.session_state.vix_color, width=2),
                                        opacity=0.9,
                                        showlegend=False
                                    ),
                                    secondary_y=False
                                )

                                fig_intraday.add_annotation(
                                    x=vix_data.index[-1],
                                    y=normalized_vix.iloc[-1],
                                    text=f"{vix_data['Close'].iloc[-1]:.2f}",
                                    showarrow=False,
                                    xshift=16,
                                    font=dict(color=st.session_state.vix_color, size=st.session_state.chart_text_size)
                                )

                                # Adjust y-axis to include VIX
                                y_min = min(y_min, normalized_vix.min() - padding)
                                y_max = max(y_max, normalized_vix.max() + padding)

                        elif st.session_state.show_vix_overlay:
                            st.warning("VIX overlay enabled but no VIX data available")

                        # Price annotation
                        if current_price is not None:
                            fig_intraday.add_annotation(
                                x=intraday_data.index[-1],
                                y=current_price,
                                xref='x',
                                yref='y',
                                xshift=27,
                                showarrow=False,
                                text=f"{current_price:,.2f}",
                                font=dict(color='yellow', size=st.session_state.chart_text_size)
                            )
                            y_min = min(y_min, current_price - padding)
                            y_max = max(y_max, current_price + padding)

                        # Process options data (GEX and DEX levels)
                        calls['OptionType'] = 'Call'
                        puts['OptionType'] = 'Put'
                        added_strikes = set()

                        # Add GEX levels if enabled
                        if st.session_state.show_gex_levels:
                            # Calculate strike range around current price (percentage-based)
                            strike_range = calculate_strike_range(current_price)
                            min_strike = current_price - strike_range
                            max_strike = current_price + strike_range
                            
                            # Filter options within strike range
                            calls_filtered = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
                            puts_filtered = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
                            
                            # Calculate Net or Absolute GEX based on gex_type setting
                            if st.session_state.gex_type == 'Net':
                                # Net GEX: Calls positive, Puts negative
                                net_gex = calls_filtered.groupby('strike')['GEX'].sum() - puts_filtered.groupby('strike')['GEX'].sum()
                            else:  # Absolute
                                # Absolute GEX: Take the larger absolute value at each strike
                                calls_gex = calls_filtered.groupby('strike')['GEX'].sum()
                                puts_gex = puts_filtered.groupby('strike')['GEX'].sum()
                                net_gex = pd.Series(index=set(calls_gex.index) | set(puts_gex.index))
                                for strike in net_gex.index:
                                    call_val = abs(calls_gex.get(strike, 0))
                                    put_val = abs(puts_gex.get(strike, 0))
                                    net_gex[strike] = call_val if call_val >= put_val else -put_val
                            
                            # Remove zero values and get top 5 by absolute value
                            net_gex = net_gex[net_gex != 0]
                            if not net_gex.empty:
                                # Create DataFrame for easier manipulation
                                gex_df = pd.DataFrame({
                                    'strike': net_gex.index,
                                    'GEX': net_gex.values,
                                    'abs_GEX': abs(net_gex.values)
                                })
                                
                                # Get top 5 by absolute GEX value
                                top5_gex = gex_df.nlargest(5, 'abs_GEX')
                                top5_gex['distance'] = abs(top5_gex['strike'] - current_price)
                                nearest_3_gex = top5_gex.nsmallest(3, 'distance')
                                max_gex = top5_gex['abs_GEX'].max()

                                for row in top5_gex.itertuples():
                                    if row.strike not in added_strikes and not pd.isna(row.GEX) and row.GEX != 0:
                                        # Calculate intensity based on GEX value relative to max
                                        intensity = max(0.6, min(1.0, row.abs_GEX / max_gex))  # Use abs_GEX for intensity
                                        
                                        # Determine color based on GEX sign (positive = call color, negative = put color)
                                        base_color = st.session_state.call_color if row.GEX >= 0 else st.session_state.put_color
                                        
                                        # Convert hex to RGB
                                        rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                                        
                                        # Create color with intensity
                                        color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {intensity})'
                                        
                                        fig_intraday.add_shape(
                                            type='line',
                                            x0=intraday_data.index[0],
                                            x1=intraday_data.index[-1],
                                            y0=row.strike,
                                            y1=row.strike,
                                            line=dict(
                                                color=color,
                                                width=2
                                            ),
                                            xref='x',
                                            yref='y',
                                            layer='below'
                                        )
                                        
                                        # Add text annotation positioned to the right of the chart
                                        gex_label = f"GEX {row.GEX:,.0f}"
                                        fig_intraday.add_annotation(
                                            x=0.92,
                                            y=row.strike,
                                            text=gex_label,
                                            font=dict(color=color, size=st.session_state.chart_text_size - 2),
                                            showarrow=False,
                                            xref="paper",  # Use paper coordinates for x
                                            yref="y",      # Use data coordinates for y
                                            xanchor="left"
                                        )
                                        added_strikes.add(row.strike)

                                # Include GEX strikes in y-axis range
                                y_min = min(y_min, nearest_3_gex['strike'].min() - padding)
                                y_max = max(y_max, nearest_3_gex['strike'].max() + padding)

                        # Add DEX levels if enabled
                        if st.session_state.show_dex_levels:
                            # Create combined DEX dataframe with absolute values for ranking
                            dex_options_df = pd.concat([calls, puts]).dropna(subset=['DEX'])
                            
                            if not dex_options_df.empty:
                                # Use absolute DEX values for ranking (similar to GEX logic)
                                dex_options_df['abs_DEX'] = abs(dex_options_df['DEX'])
                                top5_dex = dex_options_df.nlargest(5, 'abs_DEX')[['strike', 'DEX', 'OptionType']]
                                top5_dex['distance'] = abs(top5_dex['strike'] - current_price)
                                nearest_3_dex = top5_dex.nsmallest(3, 'distance')
                                max_dex = abs(top5_dex['DEX']).max()

                                for row in top5_dex.itertuples():
                                    if row.strike not in added_strikes and not pd.isna(row.DEX) and row.DEX != 0:
                                        # Calculate intensity based on DEX value relative to max
                                        intensity = max(0.6, min(1.0, abs(row.DEX) / max_dex))
                                        
                                        # Get base color from session state - use different style for DEX
                                        base_color = st.session_state.call_color if row.OptionType == 'Call' else st.session_state.put_color
                                        
                                        # Convert hex to RGB
                                        rgb = tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                                        
                                        # Create color with intensity
                                        color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {intensity})'
                                        
                                        # Add dashed line for DEX levels to distinguish from GEX
                                        fig_intraday.add_shape(
                                            type='line',
                                            x0=intraday_data.index[0],
                                            x1=intraday_data.index[-1],
                                            y0=row.strike,
                                            y1=row.strike,
                                            line=dict(
                                                color=color,
                                                width=2,
                                                dash='dash'  # Dashed line to distinguish from GEX
                                            ),
                                            xref='x',
                                            yref='y',
                                            layer='below'
                                        )
                                        
                                        # Add text annotation positioned to the right of the chart
                                        fig_intraday.add_annotation(
                                            x=0.92,
                                            y=row.strike,
                                            text=f"DEX {row.DEX:,.0f}",
                                            font=dict(color=color, size=st.session_state.chart_text_size - 2),
                                            showarrow=False,
                                            xref="paper",  # Use paper coordinates for x
                                            yref="y",      # Use data coordinates for y
                                            xanchor="left"
                                        )
                                        added_strikes.add(row.strike)

                                # Include DEX strikes in y-axis range
                                y_min = min(y_min, nearest_3_dex['strike'].min() - padding)
                                y_max = max(y_max, nearest_3_dex['strike'].max() + padding)

                        # Ensure minimum range
                        if abs(y_max - y_min) < (current_price * 0.01):  # Minimum 1% range
                            center = (y_max + y_min) / 2
                            y_min = center * 0.99
                            y_max = center * 1.01

                        # Update layout
                        fig_intraday.update_layout(
                            title=dict(
                                text=f"Intraday Price for {ticker}",
                                font=dict(size=st.session_state.chart_text_size + 4)
                            ),
                            height=600,
                            hovermode='x unified',
                            margin=dict(r=150, l=50),
                            xaxis=dict(
                                autorange=True, 
                                rangeslider=dict(visible=False),
                                showgrid=False,
                                tickfont=dict(size=st.session_state.chart_text_size)
                            ),
                            yaxis=dict(
                                autorange=True,
                                fixedrange=False,
                                showgrid=False,
                                zeroline=False,
                                tickfont=dict(size=st.session_state.chart_text_size)
                            ),
                            showlegend=bool(st.session_state.get('show_technical_indicators') and st.session_state.get('selected_indicators')),  # Show legend when technical indicators are enabled
                            legend=dict(
                                x=1.02,
                                y=1,
                                xanchor="left",
                                yanchor="top",
                                bgcolor="rgba(0,0,0,0.5)",
                                bordercolor="rgba(255,255,255,0.2)",
                                borderwidth=1
                            )
                        )


                    # Volume ratio and other charts
                    call_volume = calls['volume'].sum()
                    put_volume = puts['volume'].sum()
                    fig_volume_ratio = create_donut_chart(call_volume, put_volume)
                    fig_max_pain = create_max_pain_chart(calls, puts, S)
                    
                    chart_options = [
                        "Intraday Price", "Gamma Exposure", "Vanna Exposure", "Delta Exposure",
                        "Charm Exposure", "Speed Exposure", "Vomma Exposure", "Volume Ratio",
                        "Max Pain", "Delta-Adjusted Value Index", "Volume by Strike"
                    ]
                    default_charts = ["Intraday Price", "Gamma Exposure", "Vanna Exposure", "Delta Exposure", "Charm Exposure"]
                    selected_charts = st.multiselect("Select charts to display:", chart_options, default=[
                        chart for chart in default_charts if chart in chart_options
                    ])

                    if 'saved_ticker' in st.session_state and st.session_state.saved_ticker:
                        current_price = get_current_price(st.session_state.saved_ticker)
                        if current_price:
                            gainers_df = get_screener_data("day_gainers")
                            losers_df = get_screener_data("day_losers")
                            
                            if not gainers_df.empty and not losers_df.empty:
                                market_text = (
                                    "<span style='color: gray; font-size: 14px;'>Gainers:</span> " +
                                    " ".join([f"<span style='color: {st.session_state.call_color}'>{gainer['symbol']}: +{gainer['regularMarketChangePercent']:.1f}%</span> "
                                            for _, gainer in gainers_df.head().iterrows()]) +
                                    " | <span style='color: gray; font-size: 14px;'>Losers:</span> " +
                                    " ".join([f"<span style='color: {st.session_state.put_color}'>{loser['symbol']}: {loser['regularMarketChangePercent']:.1f}%</span> "
                                            for _, loser in losers_df.head().iterrows()])
                                )
                                st.markdown(market_text, unsafe_allow_html=True)
                            
                            # Get additional market data
                            try:
                                stock_info = yf.Ticker(st.session_state.saved_ticker).info
                                prev_close = stock_info.get('previousClose', 0)
                                day_high = stock_info.get('dayHigh', 0)
                                day_low = stock_info.get('dayLow', 0)
                                day_open = stock_info.get('regularMarketOpen', 0)
                                change = current_price - prev_close
                                change_percent = (change / prev_close) * 100
                                
                                # Get additional metrics
                                market_cap = stock_info.get('marketCap', 0)
                                market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.2f}M"
                                
                                avg_volume = stock_info.get('averageVolume', 0)
                                current_volume = stock_info.get('volume', 0)
                                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                                
                                fifty_two_week_high = stock_info.get('fiftyTwoWeekHigh', 0)
                                fifty_two_week_low = stock_info.get('fiftyTwoWeekLow', 0)
                                from_52_week_high = ((current_price - fifty_two_week_high) / fifty_two_week_high) * 100
                                from_52_week_low = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100
                                
                                # Get options data if available (to calculate call-to-put ratio)
                                call_put_ratio_text = ""
                                options_volume_text = ""
                                
                                try:
                                    stock = yf.Ticker(st.session_state.saved_ticker)
                                    if hasattr(stock, 'options') and stock.options:
                                        # Get nearest expiry
                                        nearest_expiry = get_nearest_expiry(stock.options)
                                        if nearest_expiry:
                                            calls, puts = fetch_options_for_date(st.session_state.saved_ticker, nearest_expiry, current_price)
                                            call_volume = calls['volume'].sum()
                                            put_volume = puts['volume'].sum()
                                            call_oi = calls['openInterest'].sum()
                                            put_oi = puts['openInterest'].sum()
                                            
                                            if put_volume > 0:
                                                cp_volume_ratio = call_volume / put_volume
                                                cp_ratio_color = st.session_state.call_color if cp_volume_ratio > 1 else st.session_state.put_color
                                                options_volume_text = f"<span style='color: gray;'>Call Vol:</span> <span style='color: {st.session_state.call_color}'>{call_volume:,}</span> | <span style='color: gray;'>Put Vol:</span> <span style='color: {st.session_state.put_color}'>{put_volume:,}</span>"
                                                call_put_ratio_text = f" | <span style='color: gray;'>C/P Ratio:</span> <span style='color: {cp_ratio_color}'>{cp_volume_ratio:.2f}</span>"
                                            
                                            if put_oi > 0:
                                                cp_oi_ratio = call_oi / put_oi
                                                oi_ratio_color = st.session_state.call_color if cp_oi_ratio > 1 else st.session_state.put_color
                                                call_put_ratio_text += f" | <span style='color: gray;'>OI Ratio:</span> <span style='color: {oi_ratio_color}'>{cp_oi_ratio:.2f}</span>"
                                except Exception as e:
                                    print(f"Error fetching options data: {e}")

                                # Create market data display
                                price_color = st.session_state.call_color if change >= 0 else st.session_state.put_color
                                change_symbol = '+' if change >= 0 else ''
                                
                                price_text = f"""
                                <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                                    <span style='font-size: 24px; color: {price_color}'>
                                        ${current_price:.2f} {change_symbol}{change:.2f} ({change_symbol}{change_percent:.2f}%)
                                    </span><br>
                                    <span style='color: gray; font-size: 14px;'>
                                        Open: ${day_open:.2f} | High: ${day_high:.2f} | Low: ${day_low:.2f} | Prev Close: ${prev_close:.2f}
                                    </span><br>
                                    <span style='color: gray; font-size: 14px;'>
                                        Market Cap: {market_cap_str} | Vol: {current_volume:,} ({volume_ratio:.2f}x avg)
                                    </span><br>
                                    <span style='color: gray; font-size: 14px;'>
                                        52W Range: ${fifty_two_week_low:.2f} to ${fifty_two_week_high:.2f} ({from_52_week_low:.1f}% from low, {from_52_week_high:.1f}% from high)
                                    </span>
                                    {options_volume_text and f"<br><span style='color: gray; font-size: 14px;'>{options_volume_text}{call_put_ratio_text}</span>" or ""}
                                </div>
                                """
                                st.markdown(price_text, unsafe_allow_html=True)
                            except Exception as e:
                                st.markdown(f"#### Current Price: ${current_price:.2f}")
                                print(f"Error fetching additional market data: {e}")
                            
                            st.markdown("---")
                    # Display selected charts
                    if "Intraday Price" in selected_charts:
                        st.plotly_chart(fig_intraday, use_container_width=True, key="Dashboard_intraday_chart")
                    
                    supplemental_charts = []
                    for chart, fig in [
                        ("Gamma Exposure", fig_gamma), ("Delta Exposure", fig_delta),
                        ("Vanna Exposure", fig_vanna), ("Charm Exposure", fig_charm),
                        ("Speed Exposure", fig_speed), ("Vomma Exposure", fig_vomma),
                        ("Volume Ratio", fig_volume_ratio), ("Max Pain", fig_max_pain),
                        ("Delta-Adjusted Value Index", create_davi_chart(calls, puts, S)),
                        ("Volume by Strike", create_volume_by_strike_chart(calls, puts, S))
                    ]:
                        if chart in selected_charts:
                            supplemental_charts.append(fig)
                    
                    for i in range(0, len(supplemental_charts), 2):
                        cols = st.columns(2)
                        for j, chart in enumerate(supplemental_charts[i:i+2]):
                            if chart is not None:
                                cols[j].plotly_chart(chart, use_container_width=True)

                else:
                    st.warning("Please select an expiration date to view the dashboard.")
                    st.stop()

elif st.session_state.current_page == "Max Pain":
    main_container = st.container()
    with main_container:
        st.empty()
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="max_pain_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_max_pain"):
                st.cache_data.clear()
                st.rerun()
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    lambda t, d: fetch_options_for_date(t, d, S)  # Pass S to fetch_options_for_date
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()

                # Calculate and display max pain
                result = calculate_max_pain(all_calls, all_puts)
                if result is not None:
                    max_pain_strike, call_max_pain_strike, put_max_pain_strike, *_ = result
                    st.markdown(f"### Total Max Pain Strike: ${max_pain_strike:.2f}")
                    st.markdown(f"### Call Max Pain Strike: ${call_max_pain_strike:.2f}")
                    st.markdown(f"### Put Max Pain Strike: ${put_max_pain_strike:.2f}")
                    st.markdown(f"### Current Price: ${S:.2f}")
                    st.markdown(f"### Distance to Max Pain: ${abs(S - max_pain_strike):.2f}")
                    
                    # Create and display the max pain chart
                    fig = create_max_pain_chart(all_calls, all_puts, S)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not calculate max pain point.")

if st.session_state.get('current_page') == "IV Surface":
    main_container = st.container()
    with main_container:
        # Layout for ticker input and refresh button
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input(
                "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):",
                value=st.session_state.get('saved_ticker', ''),
                key="iv_skew_ticker"
            )
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔄", key="refresh_button_skew"):
                st.cache_data.clear()
                st.rerun()

        # Format and save ticker
        ticker = format_ticker(user_ticker)
        if ticker != st.session_state.get('saved_ticker', ''):
            st.cache_data.clear()
            save_ticker(ticker)

        if ticker:
            # Fetch current price
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Get options data
            stock = yf.Ticker(ticker)
            available_dates = stock.options

            if not available_dates:
                st.warning("No options data available for this ticker.")
                st.stop()

            # Multiselect for expiration dates with no default selection
            selected_expiry_dates = st.multiselect(
                "Select Expiration Dates (1 for 2D chart, 2+ for 3D surface):",
                options=available_dates,
                default=None,  # Explicitly set to None to avoid pre-selection
                key="iv_date_selector"
            )

            # Store the user's selection
            st.session_state.iv_selected_dates = selected_expiry_dates

            # Proceed only if the user has selected at least one date
            if not selected_expiry_dates:
                st.info("Please select at least one expiration date to generate the chart.")
                st.stop()

            try:
                # Fetch options data
                with st.spinner('Fetching options data...'):
                    all_data = []  # Store all IV data

                    # Calculate strike range using percentage-based setting
                    strike_range = calculate_strike_range(S)
                    min_strike = S - strike_range
                    max_strike = S + strike_range

                    for exp_date in selected_expiry_dates:
                        expiry_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                        days_to_exp = (expiry_date - datetime.now().date()).days
                        calls, puts = fetch_options_for_date(ticker, exp_date, S)

                        # Filter strikes within range
                        calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
                        puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]

                        for strike in sorted(set(calls['strike'].unique()) | set(puts['strike'].unique())):
                            call_iv = calls[calls['strike'] == strike]['impliedVolatility'].mean()
                            put_iv = puts[puts['strike'] == strike]['impliedVolatility'].mean()
                            iv = np.nanmean([call_iv, put_iv])
                            if not np.isnan(iv):
                                all_data.append({
                                    'strike': strike,
                                    'days': days_to_exp,
                                    'iv': iv * 100  # Convert to percentage
                                })

                    if not all_data:
                        st.warning("No valid IV data available within strike range.")
                        st.stop()

                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)

                    # Create custom colorscale using call/put colors
                    call_rgb = [int(st.session_state.call_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                    put_rgb = [int(st.session_state.put_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                    custom_colorscale = [
                        [0, f'rgb({put_rgb[0]}, {put_rgb[1]}, {put_rgb[2]})'],
                        [0.5, 'rgb(255, 215, 0)'],  # Gold at center
                        [1, f'rgb({call_rgb[0]}, {call_rgb[1]}, {call_rgb[2]})']
                    ]

                    if len(selected_expiry_dates) == 1:
                        # 2D Plot
                        fig = go.Figure()

                        # Filter data for single expiration
                        single_date_df = df[df['days'] == df['days'].iloc[0]]
                        
                        # Calculate center IV for coloring
                        center_iv = single_date_df['iv'].median()

                        # Create line segments with color gradient based on IV value
                        for i in range(len(single_date_df) - 1):
                            iv_val = single_date_df['iv'].iloc[i]
                            if iv_val >= center_iv:
                                color = st.session_state.call_color
                            else:
                                color = st.session_state.put_color
                            
                            fig.add_trace(go.Scatter(
                                x=single_date_df['strike'].iloc[i:i+2],
                                y=single_date_df['iv'].iloc[i:i+2],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hovertemplate='Strike: %{x:.2f}<br>IV: %{y:.2f}%<extra></extra>'
                            ))

                        # Add current price line
                        fig.add_vline(
                            x=S,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7,
                            annotation_text=f"{S:.2f}",
                            annotation_position="top"
                        )

                        # Update layout
                        padding = strike_range * 0.05
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Implied Volatility Surface - {ticker} (Expiration: {selected_expiry_dates[0]})',
                            xaxis_title='Strike Price',
                            yaxis_title='Implied Volatility (%)',
                            yaxis=dict(tickformat='.1f', ticksuffix='%'),
                            xaxis=dict(range=[min_strike - padding, max_strike + padding]),
                            width=800,
                            height=600
                        )

                    else:
                        # 3D Surface Plot
                        # Create meshgrid for interpolation
                        unique_strikes = np.linspace(min_strike, max_strike, 200)
                        unique_days = np.linspace(df['days'].min(), df['days'].max(), 200)
                        X, Y = np.meshgrid(unique_strikes, unique_days)

                        # Interpolate surface
                        Z = griddata(
                            (df['strike'], df['days']),
                            df['iv'],
                            (X, Y),
                            method='linear',
                            fill_value=np.nan
                        )

                        # Create 3D surface plot
                        fig = go.Figure()

                        # Add IV surface with custom colorscale
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale=custom_colorscale,
                            colorbar=dict(
                                title=dict(text='IV %', side='right'),
                                tickformat='.1f',
                                ticksuffix='%'
                            ),
                            hovertemplate='Strike: %{x:.2f}<br>Days: %{y:.0f}<br>IV: %{z:.2f}%<extra></extra>'
                        ))

                        # Add current price plane
                        fig.add_trace(go.Surface(
                            x=[[S, S], [S, S]],
                            y=[[df['days'].min(), df['days'].min()], [df['days'].max(), df['days'].max()]],
                            z=[[df['iv'].min(), df['iv'].max()], [df['iv'].min(), df['iv'].max()]],
                            opacity=0.3,
                            showscale=False,
                            colorscale='oranges',
                            name='Current Price',
                            hovertemplate='Current Price: $%{x:.2f}<extra></extra>'
                        ))

                        # Update layout
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Implied Volatility Surface - {ticker}',
                            scene=dict(
                                xaxis=dict(title='Strike Price'),
                                yaxis=dict(title='Days to Expiration'),
                                zaxis=dict(title='Implied Volatility (%)', tickformat='.1f', ticksuffix='%')
                            ),
                            width=800,
                            height=800
                        )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
    st.stop()

elif st.session_state.get('current_page') == "GEX Surface":
    main_container = st.container()
    with main_container:
        # Layout for ticker input and refresh button
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input(
                "Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):",
                value=st.session_state.get('saved_ticker', ''),
                key="gex_surface_ticker"
            )
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔄", key="refresh_button_gex"):
                st.cache_data.clear()
                st.rerun()

        # Format and save ticker
        ticker = format_ticker(user_ticker)
        if ticker != st.session_state.get('saved_ticker', ''):
            st.cache_data.clear()
            save_ticker(ticker)

        if ticker:
            # Fetch current price
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            # Get options data
            stock = yf.Ticker(ticker)
            available_dates = stock.options

            if not available_dates:
                st.warning("No options data available for this ticker.")
                st.stop()

            # Multiselect for expiration dates with no default selection
            selected_expiry_dates = st.multiselect(
                "Select Expiration Dates (1 for 2D chart, 2+ for 3D surface):",
                options=available_dates,
                default=None,
                key="gex_date_selector"
            )

            # Store the user's selection
            st.session_state.gex_selected_dates = selected_expiry_dates

            # Proceed only if the user has selected at least one date
            if not selected_expiry_dates:
                st.info("Please select at least one expiration date to generate the chart.")
                st.stop()

            try:
                # Fetch options data
                with st.spinner('Fetching options data...'):
                    all_data = []  # Store all computed GEX data

                    # Calculate strike range using percentage-based setting
                    strike_range = calculate_strike_range(S)
                    min_strike = S - strike_range
                    max_strike = S + strike_range

                    for date in selected_expiry_dates:
                        # Compute greeks using the same function as gamma exposure chart
                        calls, puts, _, t, selected_expiry, today = compute_greeks_and_charts(ticker, date, "gex", S)
                        
                        if calls is not None and puts is not None:
                            days_to_exp = (selected_expiry - today).days
                            
                            # Filter and process data within strike range
                            calls = calls[(calls['strike'] >= min_strike) & (calls['strike'] <= max_strike)]
                            puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= max_strike)]
                            
                            for _, row in calls.iterrows():
                                if not pd.isna(row['GEX']) and abs(row['GEX']) >= 100:
                                    all_data.append({
                                        'strike': row['strike'],
                                        'days': days_to_exp,
                                        'gex': row['GEX']
                                    })
                            
                            for _, row in puts.iterrows():
                                if not pd.isna(row['GEX']) and abs(row['GEX']) >= 100:
                                    all_data.append({
                                        'strike': row['strike'],
                                        'days': days_to_exp,
                                        'gex': -row['GEX']
                                    })

                    if not all_data:
                        st.warning("No valid GEX data available.")
                        st.stop()

                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)

                    if len(selected_expiry_dates) == 1:
                        # 2D Plot for single expiration
                        fig = go.Figure()
                        
                        # Filter data for the single expiration date
                        single_date_df = df[df['days'] == df['days'].iloc[0]]
                        
                        # Group by strike and sum GEX values
                        grouped_gex = single_date_df.groupby('strike')['gex'].sum().reset_index()
                        
                        # Create line plot with color gradient based on GEX sign
                        for i in range(len(grouped_gex) - 1):
                            if grouped_gex['gex'].iloc[i] >= 0:
                                color = st.session_state.call_color
                            else:
                                color = st.session_state.put_color
                                
                            fig.add_trace(go.Scatter(
                                x=grouped_gex['strike'].iloc[i:i+2],
                                y=grouped_gex['gex'].iloc[i:i+2],
                                mode='lines',
                                line=dict(color=color, width=2),
                                showlegend=False,
                                hovertemplate='Strike: %{x:.2f}<br>GEX: %{y:,.0f}<extra></extra>'
                            ))
                        
                        # Add current price line
                        fig.add_vline(
                            x=S,
                            line_dash="dash",
                            line_color="white",
                            opacity=0.7,
                            annotation_text=f"{S:.2f}",
                            annotation_position="top"
                        )
                        
                        # Update layout with adjusted range
                        padding = (max_strike - min_strike) * 0.05
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Gamma Exposure Profile - {ticker} (Expiration: {selected_expiry_dates[0]})',
                            xaxis_title='Strike Price',
                            yaxis_title='Gamma Exposure',
                            width=800,
                            height=600,
                            xaxis=dict(range=[min_strike - padding, max_strike + padding])
                        )

                    else:
                        # 3D Surface Plot for multiple expirations
                        # Create meshgrid with adjusted strike range
                        padding = (max_strike - min_strike) * 0.05
                        unique_strikes = np.linspace(min_strike - padding, max_strike + padding, 200)
                        unique_days = np.linspace(df['days'].min(), df['days'].max(), 200)
                        X, Y = np.meshgrid(unique_strikes, unique_days)

                        # Aggregate GEX values by strike and days
                        df_grouped = df.groupby(['strike', 'days'])['gex'].sum().reset_index()

                        # Interpolation
                        Z = griddata(
                            (df_grouped['strike'], df_grouped['days']),
                            df_grouped['gex'],
                            (X, Y),
                            method='linear',
                            fill_value=0
                        )

                        # Create custom colorscale using call/put colors
                        call_rgb = [int(st.session_state.call_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                        put_rgb = [int(st.session_state.put_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
                        
                        colorscale = [
                            [0, f'rgb({put_rgb[0]}, {put_rgb[1]}, {put_rgb[2]})'],
                            [0.5, 'rgb(255, 215, 0)'],  # Gold at zero
                            [1, f'rgb({call_rgb[0]}, {call_rgb[1]}, {call_rgb[2]})']
                        ]

                        # Create 3D surface plot
                        fig = go.Figure()

                        # Add GEX surface with custom colorscale
                        fig.add_trace(go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale=colorscale,
                            opacity=1.0,
                            colorbar=dict(
                                title=dict(text='Net GEX', side='right'),
                                tickformat=',.0f'
                            ),
                            hovertemplate='Strike: %{x:.2f}<br>Days: %{y:.0f}<br>Net GEX: %{z:,.0f}<extra></extra>'
                        ))

                        # Add current price plane
                        fig.add_trace(go.Surface(
                            x=[[S, S], [S, S]],
                            y=[[df['days'].min(), df['days'].min()], [df['days'].max(), df['days'].max()]],
                            z=[[df['gex'].min(), df['gex'].max()], [df['gex'].min(), df['gex'].max()]],
                            opacity=0.3,
                            showscale=False,
                            colorscale='oranges',
                            name='Current Price',
                            hovertemplate='Current Price: $%{x:.2f}<extra></extra>'
                        ))

                        # Update layout
                        fig.update_layout(
                            template="plotly_dark",
                            title=f'Gamma Exposure Surface - {ticker}',
                            scene=dict(
                                xaxis=dict(title='Strike Price'),
                                yaxis=dict(title='Days to Expiration'),
                                zaxis=dict(title='Gamma Exposure', tickformat=',.0f')
                            ),
                            width=800,
                            height=800
                        )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
    st.stop()

elif st.session_state.current_page == "Analysis":
    main_container = st.container()
    with main_container:
        st.empty()
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="analysis_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_analysis"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            # Fetch 1 year of historical data for maximum analysis period
            historical_data = stock.history(period="1y", interval="1d")
            
            if historical_data.empty:
                st.warning("No historical data available for this ticker.")
                st.stop()

            # Add key price metrics at the top
            recent_data = historical_data.tail(2)
            if len(recent_data) >= 2:
                prev_close = recent_data.iloc[-2]['Close']
                current_close = recent_data.iloc[-1]['Close']
                daily_change = (current_close - prev_close) / prev_close * 100
                daily_range = recent_data.iloc[-1]['High'] - recent_data.iloc[-1]['Low']
                daily_range_pct = daily_range / current_close * 100
                
                # Add price metrics in columns
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    st.metric("Current Price", f"${S:.2f}", f"{daily_change:.2f}%")
                with metrics_cols[1]:
                    st.metric("Daily Range", f"${daily_range:.2f}", f"{daily_range_pct:.2f}%")
                with metrics_cols[2]:
                    st.metric("52W High", f"${historical_data['High'].max():.2f}")
                with metrics_cols[3]:
                    st.metric("52W Low", f"${historical_data['Low'].min():.2f}")

            # Calculate indicators with proper padding
            lookback = 20  # Standard lookback period
            padding_data = pd.concat([
                historical_data['Close'].iloc[:lookback].iloc[::-1],  # Reverse first lookback periods
                historical_data['Close']
            ])
            
            # Calculate SMA and Bollinger Bands with padding
            sma_padded = padding_data.rolling(window=lookback).mean()
            std_padded = padding_data.rolling(window=lookback).std()
            
            # Trim padding and assign to historical_data
            historical_data['SMA'] = sma_padded[lookback:].values
            historical_data['Upper Band'] = historical_data['SMA'] + 2 * std_padded[lookback:].values
            historical_data['Lower Band'] = historical_data['SMA'] - 2 * std_padded[lookback:].values

            # Calculate RSI
            def calculate_rsi(data, period=14):
                # Add padding for RSI calculation
                padding = pd.concat([data.iloc[:period].iloc[::-1], data])
                delta = padding.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                # Return only the non-padded portion
                return rsi[period:].values

            historical_data['RSI'] = calculate_rsi(historical_data['Close'])
            
            # Calculate MACD
            historical_data['EMA12'] = historical_data['Close'].ewm(span=12, adjust=False).mean()
            historical_data['EMA26'] = historical_data['Close'].ewm(span=26, adjust=False).mean()
            historical_data['MACD'] = historical_data['EMA12'] - historical_data['EMA26']
            historical_data['Signal'] = historical_data['MACD'].ewm(span=9, adjust=False).mean()
            historical_data['Histogram'] = historical_data['MACD'] - historical_data['Signal']
            
            # Calculate Historical Volatility (20-day)
            historical_data['Log_Return'] = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
            historical_data['Volatility_20d'] = historical_data['Log_Return'].rolling(window=20).std() * np.sqrt(252) * 100

            # Create technical analysis chart
            fig = make_subplots(
                rows=2, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    'Price vs. Simple Moving Average and Bollinger Bands',
                    'RSI'
                ),
                row_heights=[0.7, 0.3]
            )

            call_color = st.session_state.call_color
            put_color = st.session_state.put_color

            # Price and indicators with consistent colors
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Close'], 
                          name='Price', line=dict(color='gold')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['SMA'], 
                          name='SMA', line=dict(color='purple')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Upper Band'],
                          name='Upper Band', line=dict(color=call_color, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Lower Band'],
                          name='Lower Band', line=dict(color=put_color, dash='dash')),
                row=1, col=1
            )

            # RSI
            fig.add_trace(
                go.Scatter(x=historical_data.index,
                          y=historical_data['RSI'],
                          name='RSI',
                          line=dict(color='turquoise')),
                row=2, col=1
            )

            # Add overbought/oversold lines to the RSI chart (row 2)
            fig.add_hline(y=70, line_dash="dash", line_color=call_color,
                         row=2, col=1, annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color=put_color,
                         row=2, col=1, annotation_text="Oversold")

            # Update layout
            fig.update_layout(
                template="plotly_dark",
                title=dict(
                    text=f"Technical Analysis for {ticker}",
                    x=0,
                    xanchor='left',
                    font=dict(size=st.session_state.chart_text_size + 8)
                ),
                showlegend=True,
                height=800,  # Reduced to better fit with weekday returns
                legend=dict(
                    font=dict(size=st.session_state.chart_text_size)
                )
            )

            # Update axes for both rows
            for i in range(1, 3):
                fig.update_xaxes(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size),
                    row=i, col=1
                )
                fig.update_yaxes(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size),
                    row=i, col=1
                )

            # Set y-axis range for RSI
            fig.update_yaxes(range=[0, 100], row=2, col=1)

            # Display technical analysis chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Create MACD chart
            macd_fig = make_subplots(rows=1, cols=1)
            
            macd_fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['MACD'], 
                          name='MACD', line=dict(color='blue')),
            )
            macd_fig.add_trace(
                go.Scatter(x=historical_data.index, y=historical_data['Signal'], 
                          name='Signal', line=dict(color='red')),
            )
            
            # Add histogram as bar chart
            macd_fig.add_trace(
                go.Bar(x=historical_data.index, y=historical_data['Histogram'],
                      name='Histogram',
                      marker_color=historical_data['Histogram'].apply(
                          lambda x: call_color if x >= 0 else put_color
                      )),
            )
            
            macd_fig.update_layout(
                title=dict(
                    text="MACD Indicator",
                    x=0,
                    xanchor='left',
                    font=dict(size=st.session_state.chart_text_size + 4)
                ),
                height=300,
                template="plotly_dark",
                legend=dict(
                    font=dict(size=st.session_state.chart_text_size)
                ),
                xaxis=dict(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size)
                ),
                yaxis=dict(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size)
                )
            )
            
            # Display MACD chart
            st.plotly_chart(macd_fig, use_container_width=True)
            
            # Historical volatility chart
            vol_fig = go.Figure()
            vol_fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Volatility_20d'],
                    name='20-Day HV',
                    line=dict(color='orange', width=2)
                )
            )
            
            vol_fig.update_layout(
                title=dict(
                    text="Historical Volatility (20-Day)",
                    x=0,
                    xanchor='left',
                    font=dict(size=st.session_state.chart_text_size + 4)
                ),
                height=300,
                template="plotly_dark",
                xaxis=dict(
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size)
                ),
                yaxis=dict(
                    title="Volatility %",
                    tickfont=dict(size=st.session_state.chart_text_size),
                    title_font=dict(size=st.session_state.chart_text_size),
                    ticksuffix="%"
                )
            )
            
            # Display volatility chart
            st.plotly_chart(vol_fig, use_container_width=True)

            # Add trend indicator section
            st.subheader("Technical Indicators Summary")
            
            # Calculate trend indicators
            current_price = historical_data['Close'].iloc[-1]
            sma20 = historical_data['SMA'].iloc[-1]
            sma50 = historical_data['Close'].rolling(window=50).mean().iloc[-1]
            sma200 = historical_data['Close'].rolling(window=200).mean().iloc[-1]
            rsi = historical_data['RSI'].iloc[-1]
            macd = historical_data['MACD'].iloc[-1]
            signal = historical_data['Signal'].iloc[-1]
            
            # Create indicator cards in columns
            indicator_cols = st.columns(3)
            
            with indicator_cols[0]:
                st.markdown("**Price vs Moving Averages**")
                ma_indicators = [
                    f"Price vs 20 SMA: {'Bullish' if current_price > sma20 else 'Bearish'}",
                    f"Price vs 50 SMA: {'Bullish' if current_price > sma50 else 'Bearish'}",
                    f"Price vs 200 SMA: {'Bullish' if current_price > sma200 else 'Bearish'}"
                ]
                for ind in ma_indicators:
                    st.markdown(f"- {ind}")
            
            with indicator_cols[1]:
                st.markdown("**Momentum Indicators**")
                momentum_indicators = [
                    f"RSI (14): {rsi:.2f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})",
                    f"MACD: {macd:.4f} vs Signal: {signal:.4f}",
                    f"MACD Signal: {'Bullish' if macd > signal else 'Bearish'}"
                ]
                for ind in momentum_indicators:
                    st.markdown(f"- {ind}")
            
            with indicator_cols[2]:
                st.markdown("**Volatility Indicators**")
                recent_vol = historical_data['Volatility_20d'].iloc[-1]
                avg_vol = historical_data['Volatility_20d'].mean()
                vol_indicators = [
                    f"Current HV (20d): {recent_vol:.2f}%",
                    f"Avg HV (1yr): {avg_vol:.2f}%",
                    f"Vol Trend: {'Above Average' if recent_vol > avg_vol else 'Below Average'}"
                ]
                for ind in vol_indicators:
                    st.markdown(f"- {ind}")

            # Add weekday returns analysis without extra spacing
            st.subheader("Weekday Returns Analysis")
            
            period = st.selectbox(
                "Select Analysis Period:",
                options=['1y', '6mo', '3mo', '1mo'], 
                format_func=lambda x: {
                    '1y': '1 Year',
                    '6mo': '6 Months', 
                    '3mo': '3 Months',
                    '1mo': '1 Month'
                }[x],
                key="weekday_returns_period"
            )

            weekday_returns = calculate_annualized_return(historical_data, period)
            weekday_fig = create_weekday_returns_chart(weekday_returns)
            st.plotly_chart(weekday_fig, use_container_width=True)

    st.stop()

elif st.session_state.current_page == "Delta-Adjusted Value Index":
    main_container = st.container()
    with main_container:
        st.empty()
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="davi_ticker")
        with col2:
            st.write("")
            st.write("")
            if st.button("🔄", key="refresh_button_davi"):
                st.cache_data.clear()
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.warning("Please select at least one expiration date.")
                    st.stop()
                
                # The issue is here - we need to make sure the Greek values are computed
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker,
                    selected_expiry_dates,
                    # Use compute_greeks_and_charts to ensure calc_delta is calculated
                    lambda t, d: compute_greeks_and_charts(t, d, "davi", S)[:2]
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                    
                # Calculate delta values if they don't exist
                if 'calc_delta' not in all_calls.columns:
                    all_calls = all_calls.copy()
                    all_puts = all_puts.copy()
                    
                    # Calculate days to expiry for each option
                    today = datetime.today().date()
                    all_calls['t'] = (all_calls['extracted_expiry'] - today).dt.days / 365.0
                    all_puts['t'] = (all_puts['extracted_expiry'] - today).dt.days / 365.0
                    
                    # Define delta calculation function
                    def compute_delta(row, flag):
                        try:
                            sigma = row.get("impliedVolatility", None)
                            if sigma is None or sigma <= 0:
                                return None
                            t = row.get("t", None)
                            if t is None or t <= 0:
                                return None
                            K = row.get("strike", None)
                            if K is None:
                                return None
                            return calculate_greeks(flag, S, K, t, sigma)['delta']
                        except Exception:
                            return None
                    
                    # Calculate delta
                    all_calls['calc_delta'] = all_calls.apply(lambda row: compute_delta(row, "c"), axis=1)
                    all_puts['calc_delta'] = all_puts.apply(lambda row: compute_delta(row, "p"), axis=1)

                fig = create_davi_chart(all_calls, all_puts, S)
                st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_page == "Implied Probabilities":
    main_container = st.container()
    with main_container:
        st.empty()  # Clear previous content
        
        # Header
        st.title("🎲 Implied Probabilities Analysis")
        st.markdown("""
        **Analyze option-implied probabilities and expected moves based on market pricing.**
        
        This page calculates:
        - **16 Delta levels (1σ)** - Industry standard one standard deviation levels (16%/84% probability)
        - **30 Delta levels** - Common institutional trading levels (30%/70% probability)
        - **Implied move** - Expected price range based on straddle pricing
        - **Probability distribution** - Market-implied likelihood of price levels using Black-Scholes
        - **Trading ranges** - Expected breakout levels and support/resistance zones
        """)
        
        col1, col2 = st.columns([0.94, 0.06])
        with col1:
            user_ticker = st.text_input("Enter Stock Ticker (e.g., SPY, TSLA, SPX, NDX):", saved_ticker, key="implied_prob_ticker")
        with col2:
            st.write("")  # Add some spacing
            st.write("")  # Add some spacing
            if st.button("🔄", key="refresh_button_implied_prob"):
                st.cache_data.clear()  # Clear the cache before rerunning
                st.rerun()
        
        ticker = format_ticker(user_ticker)
        
        # Clear cache if ticker changes
        if ticker != saved_ticker:
            st.cache_data.clear()
            save_ticker(ticker)  # Save the ticker
        
        if ticker:
            # Fetch price once
            S = get_current_price(ticker)
            if S is None:
                st.error("Could not fetch current price.")
                st.stop()

            stock = yf.Ticker(ticker)
            available_dates = stock.options
            if not available_dates:
                st.warning("No options data available for this ticker.")
            else:
                selected_expiry_dates, selector_container = expiry_selector_fragment(st.session_state.current_page, available_dates)
                st.session_state.expiry_selector_container = selector_container
                
                if not selected_expiry_dates:
                    st.info("Please select at least one expiration date.")
                    st.stop()
                
                # For implied probabilities, we typically focus on the nearest expiry
                # But allow multiple for comparison
                all_calls, all_puts = fetch_and_process_multiple_dates(
                    ticker, 
                    selected_expiry_dates,
                    lambda t, d: compute_greeks_and_charts(t, d, "implied_prob", S)[:2]  # Only take calls and puts
                )
                
                if all_calls.empty and all_puts.empty:
                    st.warning("No options data available for the selected dates.")
                    st.stop()
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Delta Levels", "📈 Probability Charts", "📋 Detailed Analysis"])
                
                with tab1:
                    st.subheader("Key Probability Metrics")
                    
                    # Calculate key metrics for the first (nearest) expiry
                    nearest_expiry = selected_expiry_dates[0]
                    nearest_calls = all_calls[all_calls['extracted_expiry'] == pd.to_datetime(nearest_expiry).date()]
                    nearest_puts = all_puts[all_puts['extracted_expiry'] == pd.to_datetime(nearest_expiry).date()]
                    
                    # Calculate implied move
                    implied_move_data = calculate_implied_move(S, nearest_calls, nearest_puts)
                    
                    # Calculate delta-based probability strikes (industry standard)
                    prob_16_data = find_probability_strikes(nearest_calls, nearest_puts, S, nearest_expiry, 0.16)  # ~1 standard deviation
                    prob_30_data = find_probability_strikes(nearest_calls, nearest_puts, S, nearest_expiry, 0.30)  # Common institutional level
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${S:.2f}")
                        if implied_move_data:
                            st.metric("Implied Move", f"${implied_move_data['implied_move_dollars']:.2f}", 
                                     f"({implied_move_data['implied_move_pct']:.1f}%)")
                    
                    with col2:
                        if prob_16_data and prob_16_data['strike_above']:
                            st.metric("16Δ Above (1σ) - 16%", f"${prob_16_data['strike_above']:.2f}", 
                                     f"Actual: {prob_16_data['prob_above']*100:.1f}% above")
                        if prob_16_data and prob_16_data['strike_below']:
                            st.metric("16Δ Below (1σ) - 84%", f"${prob_16_data['strike_below']:.2f}", 
                                     f"Actual: {prob_16_data['prob_below']*100:.1f}% below")
                    
                    with col3:
                        if prob_30_data and prob_30_data['strike_above']:
                            st.metric("30Δ Above - 30%", f"${prob_30_data['strike_above']:.2f}", 
                                     f"Actual: {prob_30_data['prob_above']*100:.1f}% above")
                        if prob_30_data and prob_30_data['strike_below']:
                            st.metric("30Δ Below - 70%", f"${prob_30_data['strike_below']:.2f}", 
                                     f"Actual: {prob_30_data['prob_below']*100:.1f}% below")
                    
                    # Expected trading range with better formatting
                    if implied_move_data:
                        st.subheader("📊 Expected Trading Range")
                        
                        range_col1, range_col2, range_col3 = st.columns(3)
                        with range_col1:
                            st.metric("Lower Range", f"${implied_move_data['lower_range']:.2f}")
                        with range_col2:
                            st.metric("Upper Range", f"${implied_move_data['upper_range']:.2f}")
                        with range_col3:
                            range_width = implied_move_data['upper_range'] - implied_move_data['lower_range']
                            range_pct = (range_width / S) * 100
                            st.metric("Range Width", f"${range_width:.2f}", f"({range_pct:.2f}%)")
                    
                    # Probability levels relative to current price with better formatting
                    st.subheader("🎯 Price Targets & Movement Required")
                    
                    # Create tabs for different delta levels (industry standard)
                    prob_tab1, prob_tab2 = st.tabs(["16Δ Levels (1σ) - 16%/84%", "30Δ Levels - 30%/70%"])
                    
                    with prob_tab1:
                        if prob_16_data and prob_16_data['strike_above'] and prob_16_data['strike_below']:
                            strike_16_above = prob_16_data['strike_above']
                            strike_16_below = prob_16_data['strike_below']
                            
                            # Ensure proper ordering
                            if strike_16_above < strike_16_below:
                                strike_16_above, strike_16_below = strike_16_below, strike_16_above
                            
                            distance_above = strike_16_above - S
                            distance_below = strike_16_below - S
                            distance_pct_above = (distance_above / S) * 100
                            distance_pct_below = (distance_below / S) * 100
                            
                            # Upper and lower bounds
                            bound_col1, bound_col2 = st.columns(2)
                            
                            with bound_col1:
                                st.markdown("**🔺 16Δ Upper Level (1σ) - 16% Above**")
                                st.metric("Strike Price", f"${strike_16_above:.2f}")
                                st.metric("Distance", f"${distance_above:.2f}", f"⬆️ {distance_pct_above:+.2f}%")
                                
                            with bound_col2:
                                st.markdown("**🔻 16Δ Lower Level (1σ) - 84% Above**")
                                st.metric("Strike Price", f"${strike_16_below:.2f}")
                                st.metric("Distance", f"${abs(distance_below):.2f}", f"⬇️ {distance_pct_below:+.2f}%")
                            
                            # Range analysis
                            st.markdown("**📏 Range Analysis**")
                            range_col1, range_col2, range_col3 = st.columns(3)
                            
                            with range_col1:
                                range_width = strike_16_above - strike_16_below
                                st.metric("Total Range", f"${range_width:.2f}")
                            
                            with range_col2:
                                range_pct = (range_width / S) * 100
                                st.metric("Range %", f"{range_pct:.2f}%")
                            
                            with range_col3:
                                # Calculate if current price is within the range
                                if strike_16_below <= S <= strike_16_above:
                                    position = "Within Range ✅"
                                elif S > strike_16_above:
                                    position = "Above Range ⬆️"
                                else:
                                    position = "Below Range ⬇️"
                                st.metric("Current Position", position)
                            
                            # Explanation
                            st.info(f"💡 **16 Delta Levels (1σ) - 16%/84%**: These represent approximate one standard deviation levels. The upper level has a 16% probability above, lower level has 84% probability above. There's roughly a 68% probability the stock will finish between ${strike_16_below:.2f} and ${strike_16_above:.2f} at expiration.")
                    
                    with prob_tab2:
                        if prob_30_data and prob_30_data['strike_above'] and prob_30_data['strike_below']:
                            strike_30_above = prob_30_data['strike_above']
                            strike_30_below = prob_30_data['strike_below']
                            
                            # Ensure proper ordering
                            if strike_30_above < strike_30_below:
                                strike_30_above, strike_30_below = strike_30_below, strike_30_above
                            
                            distance_above = strike_30_above - S
                            distance_below = strike_30_below - S
                            distance_pct_above = (distance_above / S) * 100
                            distance_pct_below = (distance_below / S) * 100
                            
                            # Upper and lower bounds
                            bound_col1, bound_col2 = st.columns(2)
                            
                            with bound_col1:
                                st.markdown("**🔺 30Δ Upper Level - 30% Above**")
                                st.metric("Strike Price", f"${strike_30_above:.2f}")
                                st.metric("Distance", f"${distance_above:.2f}", f"⬆️ {distance_pct_above:+.2f}%")
                                
                            with bound_col2:
                                st.markdown("**🔻 30Δ Lower Level - 70% Above**")
                                st.metric("Strike Price", f"${strike_30_below:.2f}")
                                st.metric("Distance", f"${abs(distance_below):.2f}", f"⬇️ {distance_pct_below:+.2f}%")
                            
                            # Range analysis
                            st.markdown("**📏 Range Analysis**")
                            range_col1, range_col2, range_col3 = st.columns(3)
                            
                            with range_col1:
                                range_width = strike_30_above - strike_30_below
                                st.metric("Total Range", f"${range_width:.2f}")
                            
                            with range_col2:
                                range_pct = (range_width / S) * 100
                                st.metric("Range %", f"{range_pct:.2f}%")
                            
                            with range_col3:
                                # Calculate if current price is within the range
                                if strike_30_below <= S <= strike_30_above:
                                    position = "Within Range ✅"
                                elif S > strike_30_above:
                                    position = "Above Range ⬆️"
                                else:
                                    position = "Below Range ⬇️"
                                st.metric("Current Position", position)
                            
                            # Explanation
                            st.info(f"💡 **30 Delta Levels - 30%/70%**: Common institutional trading levels. The upper level has a 30% probability above, lower level has 70% probability above. There's roughly a 40% probability the stock will finish between ${strike_30_below:.2f} and ${strike_30_above:.2f} at expiration.")
                
                with tab2:
                    st.subheader("Probability Levels Analysis")
                    
                    # Create a detailed table of probability levels
                    prob_levels = [0.10, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.84, 0.90]
                    prob_data = []
                    
                    for prob in prob_levels:
                        prob_info = find_probability_strikes(nearest_calls, nearest_puts, S, nearest_expiry, prob)
                        if prob_info:
                            prob_data.append({
                                'Probability Level': f"{prob*100:.0f}%",
                                'Strike Above': f"${prob_info['strike_above']:.2f}" if prob_info['strike_above'] else "N/A",
                                'Strike Below': f"${prob_info['strike_below']:.2f}" if prob_info['strike_below'] else "N/A",
                                'Actual Prob Above': f"{prob_info['prob_above']*100:.1f}%" if prob_info['prob_above'] else "N/A",
                                'Actual Prob Below': f"{prob_info['prob_below']*100:.1f}%" if prob_info['prob_below'] else "N/A"
                            })
                    
                    if prob_data:
                        prob_df_display = pd.DataFrame(prob_data)
                        st.dataframe(prob_df_display, use_container_width=True)
                    
                    # Explanatory text
                    st.markdown("""
                    **Understanding Probability Levels:**
                    - **50% Probability**: Even odds - coin flip probability of being above/below
                    - **70% Probability**: High confidence levels for directional moves
                    - **16% Probability**: Approximately one standard deviation (84% chance of staying within range)
                    - **84% Probability**: Very high confidence levels, approximately one standard deviation
                    - **90% Probability**: Extreme confidence levels for range-bound strategies
                    
                    **Key Insight:** The wider the gap between "Strike Above" and "Strike Below" for the same probability, 
                    the higher the implied volatility and expected price movement.
                    """)
                
                with tab3:
                    st.subheader("Probability Visualization")
                    
                    # Calculate probability distribution
                    prob_df = calculate_probability_distribution(nearest_calls, nearest_puts, S, nearest_expiry)
                    
                    # Create comprehensive chart
                    if not prob_df.empty:
                        fig = create_implied_probabilities_chart(prob_df, S, prob_16_data, prob_30_data, implied_move_data)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not calculate probability distribution.")
                
                with tab4:
                    st.subheader("Detailed Probability Analysis")
                    
                    # Show probability distribution table
                    if not prob_df.empty:
                        # Add additional calculations
                        prob_df['distance_from_current'] = abs(prob_df['strike'] - S)
                        prob_df['prob_above_pct'] = prob_df['prob_above'] * 100
                        prob_df['prob_below_pct'] = prob_df['prob_below'] * 100
                        
                        # Format for display
                        display_df = prob_df[['strike', 'prob_above_pct', 'prob_below_pct', 'distance_from_current']].copy()
                        display_df.columns = ['Strike', 'Prob Above (%)', 'Prob Below (%)', 'Distance from Current']
                        display_df['Strike'] = display_df['Strike'].apply(lambda x: f"${x:.2f}")
                        display_df['Prob Above (%)'] = display_df['Prob Above (%)'].apply(lambda x: f"{x:.1f}%")
                        display_df['Prob Below (%)'] = display_df['Prob Below (%)'].apply(lambda x: f"{x:.1f}%")
                        display_df['Distance from Current'] = display_df['Distance from Current'].apply(lambda x: f"${x:.2f}")
                        
                        st.dataframe(display_df, use_container_width=True)
                    
                    # Additional metrics
                    if implied_move_data:
                        st.subheader("Implied Move Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**ATM Straddle Analysis:**")
                            st.write(f"ATM Strike: ${implied_move_data['atm_strike']:.2f}")
                            st.write(f"Straddle Price: ${implied_move_data['straddle_price']:.2f}")
                            st.write(f"Implied Move: {implied_move_data['implied_move_pct']:.2f}%")
                        
                        with col2:
                            st.write("**Breakeven Levels:**")
                            st.write(f"Upper Breakeven: ${implied_move_data['upper_range']:.2f}")
                            st.write(f"Lower Breakeven: ${implied_move_data['lower_range']:.2f}")
                            
                            # Calculate probability of staying within range
                            if not prob_df.empty:
                                within_range = prob_df[
                                    (prob_df['strike'] >= implied_move_data['lower_range']) & 
                                    (prob_df['strike'] <= implied_move_data['upper_range'])
                                ]
                                if not within_range.empty:
                                    prob_within = within_range['prob_above'].iloc[-1] - within_range['prob_above'].iloc[0]
                                    st.write(f"Prob within range: {prob_within*100:.1f}%")
                    
                    st.markdown("""
                    **Note:** Probabilities are derived from option delta values and implied volatilities. 
                    These represent the market's implied view of future price movements, not predictions.
                    """)

# -----------------------------------------
# Auto-refresh
# -----------------------------------------
# Check if we're on the OI & Volume page and if market maker data has been fetched
is_market_maker_active = (
    st.session_state.current_page == "OI & Volume" and
    st.session_state.get('mm_data') is not None
)

# Only auto-refresh if not on market maker tab with active data
if not is_market_maker_active:
    refresh_rate = float(st.session_state.get('refresh_rate', 10))  # Convert to float
    if not st.session_state.get("loading_complete", False):
        st.session_state.loading_complete = True
        st.rerun()
    else:
        time.sleep(refresh_rate)
        st.rerun()

def calculate_heikin_ashi(df):
    """Calculate Heikin Ashi candlestick values."""
    ha_df = pd.DataFrame(index=df.index)
    
    # Calculate Heikin Ashi values
    ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    
    # Initialize HA_Open with first candle's opening price
    ha_df['HA_Open'] = pd.Series(index=df.index)
    ha_df.loc[ha_df.index[0], 'HA_Open'] = df['Open'].iloc[0]
    
    # Calculate subsequent HA_Open values
    for i in range(1, len(df)):
        ha_df.loc[ha_df.index[i], 'HA_Open'] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2
    
    ha_df['HA_High'] = df[['High', 'Open', 'Close']].max(axis=1)
    ha_df['HA_Low'] = df[['Low', 'Open', 'Close']].min(axis=1)
    
    return ha_df