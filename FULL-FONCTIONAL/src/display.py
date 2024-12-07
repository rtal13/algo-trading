import matplotlib.pyplot
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

def print_status(is_ok):
    """
    Prints [OK] in green if the input is True, and [KO] in red if the input is False.
    """
    # ANSI escape codes for colors
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    if is_ok:
        print(f"{GREEN}[OK]{RESET}")
    else:
        print(f"{RED}[KO]{RESET}")

def colored_print(text, color='blue'):
    # Define a dictionary of color codes
    colors = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37'
    }

    # Get the color code, defaulting to blue if not found
    color_code = colors.get(color.lower(), '34')

    RESET = '\033[0m'  # Reset color code
    print(f"\033[{color_code}m{text}{RESET}")
    
def execute_with_status(title: str, func: callable, *args, **kwargs):
    """
    Executes a given function and prints a title with a status message.
    
    Args:
        title (str): The title to print before execution.
        func (callable): The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    
    Returns:
        Any: The result of the function if executed successfully.
    """
    try:
        # Print the title in yellow
        print(colored_print(title, "yellow"))
        
        # Execute the function with provided arguments
        result = func(*args, **kwargs)
        
        # Print success status in green
        print(colored_print("✔ Success", "green"))
        return result
    except Exception as e:
        # Print failure status in red with the error message
        print(colored_print(f"✘ Failed: {str(e)}", "red"))
        return None

from tabulate import tabulate

def print_indicators(df):
    poly_cols = [col for col in df.columns if col.startswith('poly_')]

    # Define categories and their associated columns
    indicator_categories = {
        'Trend Indicators': ['RSI', 'MACD', 'EMA', 'CCI', 'ADX'],
        'Momentum Indicators': ['ROC', 'TSI', 'UO'],
        'Volume Indicators': ['CMF', 'VO'],
        'Volatility Indicators': ['ATR', 'DC_H', 'DC_L', 'DC_M', 'DC_Width'],
        'Additional Indicators': ['ICHIMOKU_A', 'ICHIMOKU_B', 'PSAR', 'VWAP', 'PP', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3'],
    }

    # Filter to ensure we only list columns that actually exist in df
    filtered_categories = {
        cat: [c for c in cols if c in df.columns]
        for cat, cols in indicator_categories.items()
        if cols  # Only include categories that have been defined
    }

    # Prepare data for tabulate: one row per category
    table_data = []
    for category, cols in filtered_categories.items():
        table_data.append([category, ", ".join(cols)])

    # Print the table
    print(tabulate(table_data, headers=["Indicator Type", "Columns"], tablefmt="psql"))

def display_load_data(df):
    stats = [
        ["Data Length", len(df)],
        ["Highest Value", df["Close"].max()],
        ["Lowest Value", df["Close"].min()],
        ["Mean Value", df["Close"].mean()],
        ["Standard Deviation", df["Close"].std()],
        ["Median Value", df["Close"].median()],
        ["Mode Value", df["Close"].mode()[0]],
        ["Variance Value", df["Close"].var()],
        ["Skewness Value", df["Close"].skew()],
        ["Kurtosis Value", df["Close"].kurt()],
    ]

    # Check for missing data
    missing_data = df.isnull().sum().to_frame(name="Missing Count").reset_index()
    missing_data.columns = ["Column", "Missing Count"]

    # Create tables using tabulate
    stats_table = tabulate(stats, headers=["Statistic", "Value"], tablefmt="grid")
    missing_data_table = tabulate(missing_data, headers="keys", tablefmt="grid")

    # Print the results
    print(stats_table)
    # print("\nMISSING DATA:")
    # print(missing_data_table)
    
def plot_indicator_categories(df, layout=(2,2), figsize=(12, 8), categories=None):
    if categories is None:
        categories = {
        'trend': ['EMA', 'MACD', 'CCI', 'ADX'],
        'momentum': ['RSI', 'ROC', 'TSI', 'UO'],
        'volume': ['CMF', 'VO'],
        'volatility': ['ATR', 'DC_H', 'DC_L', 'DC_M']
        }
    
    # Ensure the number of categories fits into the layout
    n_cats = len(categories)
    rows, cols = layout
    if n_cats > rows * cols:
        raise ValueError("Not enough subplots for all categories. Increase layout size.")
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # If layout is just one row or one column, axes might not be a 2D array
    # Normalize 'axes' to always be a 2D list for consistent indexing
    if rows == 1 and cols == 1:
        # Only one subplot
        axes = [[axes]]
    elif rows == 1:
        # One row, multiple columns
        axes = [axes]
    elif cols == 1:
        # One column, multiple rows
        axes = [[ax] for ax in axes]
    
    # Flatten category list so we can iterate easily
    cat_items = list(categories.items())
    
    # Plot each category in a separate subplot
    for idx, (cat_name, indicators) in enumerate(cat_items):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        
        # Filter indicators that are actually in df
        plot_cols = [col for col in indicators if col in df.columns]
        
        if not plot_cols:
            # If no valid columns, just note it on the plot
            ax.text(0.5, 0.5, 'No valid indicators to plot', 
                    ha='center', va='center', transform=ax.transAxes)
        else:
            # Plot each valid indicator
            for col in plot_cols:
                ax.plot(df.index, df[col], label=col)
            
            # Add a legend if we plotted something
            ax.legend()
        
        # Set title
        ax.set_title(f"{cat_name.capitalize()} Indicators")
        ax.grid(True)
    
    # Hide any unused subplots if n_cats < rows*cols
    total_plots = rows * cols
    if n_cats < total_plots:
        for extra_idx in range(n_cats, total_plots):
            r = extra_idx // cols
            c = extra_idx % cols
            axes[r][c].axis('off')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()