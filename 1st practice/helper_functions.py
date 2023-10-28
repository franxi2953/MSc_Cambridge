import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
from scipy.fftpack import fft

def combine_csv_files_in_path(path):
    """
    Combine data from multiple CSV files present in the specified directory into a single dataframe.
    
    Parameters:
    - path (str): Path to the directory containing the CSV files.
    
    Returns:
    - main_df (pd.DataFrame): A combined dataframe with data from all CSVs.
    
    Note:
    1. The function assumes that each CSV contains a single column of data.
    2. The column in the combined dataframe will be named after the CSV filename.
    3. If a CSV has more than one column, it will be skipped with a printed warning.
    4. The function assumes 2 milliseconds between each data point and assigns this as the index.
    """

    # Get a list of all CSV files in the specified path
    all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]
    
    # Initialize an empty dataframe
    main_df = pd.DataFrame()

    # Iterate through each file and add its contents to the main dataframe
    for file in all_files:
        # Load the CSV file into a dataframe
        df = pd.read_csv(os.path.join(path, file))
        
        # Ensure the CSV only contains one column
        if df.shape[1] != 1:
            print(f"File {file} has more than one column. Skipping...")
            continue
        
        # Rename the column to the filename (or any other naming logic you prefer)
        col_name = os.path.splitext(file)[0]
        df = df.rename(columns={df.columns[0]: col_name})

        # Join the current CSV's dataframe with the main dataframe
        main_df = main_df.join(df, how='outer')

    num_rows = main_df.shape[0]
    main_df.index = np.arange(2, 2 * num_rows + 1, 2)
    main_df.index.name = "milliseconds"
    
    return main_df

    color_dict = {}
    for i, col in enumerate(columns):
        color_dict[col] = color_scale[i % len(color_scale)]
    return color_dict

def plot_raw_and_normalized(df, columns_to_plot=None,plot_output=True):
    """
    Plot raw data and its corresponding min-max normalized data side by side.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe containing raw data.
    - columns_to_plot (list or range, optional): A list of column names to plot or a range.
        If None, plots all columns. Default is None.
    
    Returns:
    - normalized_df (pd.DataFrame): A dataframe containing the min-max normalized data.
    
    Note: The function also displays a plot but doesn't return the plot object.
    """

    # Selecting columns to plot
    if columns_to_plot is None:
        col_names = df.columns.tolist()
    elif isinstance(columns_to_plot, range):
        col_names = df.columns[columns_to_plot].tolist()
    else:
        col_names = columns_to_plot

    n_columns = len(col_names)
    colors = cycle(go.Figure().full_figure_for_development()['layout']['colorway'])
    fig = make_subplots(rows=n_columns, cols=2, shared_xaxes=True, subplot_titles=[name for pair in col_names for name in (pair, f"{pair} (Normalized)")])

    # Min-max normalization function
    def min_max_normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    normalized_df = pd.DataFrame(index=df.index)

    # Plotting columns
    for i, col_name in enumerate(col_names):
        current_color = next(colors)
        fig.add_trace(go.Scatter(x=df.index, y=df[col_name], mode='lines', name=col_name, line=dict(color=current_color)), row=i+1, col=1)
        normalized_data = min_max_normalize(df[col_name])
        normalized_df[col_name + '_normalized'] = normalized_data  # Add normalized data to new dataframe
        fig.add_trace(go.Scatter(x=df.index, y=normalized_data, mode='lines', name=f"{col_name} (Normalized)", line=dict(color=current_color)), row=i+1, col=2)

        # Adding the red shaded region
        fig.add_shape(
            go.layout.Shape(
                type="rect",
                x0=0,
                x1=500,
                y0=0,
                y1=1,
                fillcolor="red",
                opacity=0.3,
                layer="below",
                line_width=0,
            ),
            row=i+1,
            col=2
        )

    # Update layout
    fig.update_layout(title='', xaxis_title='Milliseconds', plot_bgcolor='white', paper_bgcolor='white', width=1400, height=4000*n_columns/15, showlegend=False)

    global_min = df.min().min()
    global_max = df.max().max()
    for i in range(1, n_columns+1):
        fig.update_yaxes(range=[global_min, global_max], row=i, col=1)
        fig.update_yaxes(range=[0, 1], row=i, col=2)
    
    if plot_output:
        fig.show()
    
    return normalized_df

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip('#')
    h_length = len(hex_color)
    rgba = tuple(int(hex_color[i:i + h_length // 3], 16) for i in range(0, h_length, h_length // 3))
    return rgba + (alpha, )

def box_plot_dataframe(df,plot_output=True):
    """
    Create box plots for the first 500ms of each column in the DataFrame, 
    representing standard variance and median, with additional lines 
    for the average of the 5% highest and lowest values of the entire column.
    
    Parameters:
    - df: DataFrame containing data to plot. Each row is a 2ms increment.
    
    Returns:
    - fig: plotly.graph_objects Figure containing the box plots.
    - results_df: DataFrame containing the 2 whiskers and 2 limits for each boxplot.
    """
    # Calculate number of rows that correspond to 500ms (since each row is 2ms)
    num_rows = int(500 / 2)
    
    # Select the first 500ms of data
    subset_df = df.iloc[:num_rows]

    # Create the color cycle for plotting
    colors = cycle(go.Figure().full_figure_for_development()['layout']['colorway'])
    
    fig = go.Figure()
    results = []

    for index, col_name in enumerate(subset_df.columns):
        # Calculate the number of values to consider for the whiskers
        n_values = int(0.05 * len(df))
        
        lower_values = df[col_name].nsmallest(n_values)
        upper_values = df[col_name].nlargest(n_values)

        # Using a loop to calculate the mean for lower and upper whisker
        lower_sum, upper_sum = 0, 0
        for l_val, u_val in zip(lower_values, upper_values):
            lower_sum += l_val
            upper_sum += u_val

        lower_whisker = lower_sum / n_values
        upper_whisker = upper_sum / n_values

        median_value = subset_df[col_name].median()
        std_value = subset_df[col_name].std()

        lower_limit = subset_df[col_name].min()
        upper_limit = subset_df[col_name].max()
        
        results.append({
            'Column': col_name,
            'Lower Whisker': lower_whisker,
            'Upper Whisker': upper_whisker,
            'Lower Limit': lower_limit,
            'Upper Limit': upper_limit
        })
        
        current_color = next(colors)
        translucent_color = 'rgba' + str(hex_to_rgba(current_color, 0.5))

        fig.add_trace(go.Box(
            y=subset_df[col_name],
            name=col_name,
            boxpoints=False,
            line_color=current_color,
            fillcolor=translucent_color,
            whiskerwidth=0
        ))

        fig.add_shape(type="line",
                      x0=index - 0.4, x1=index + 0.4,
                      y0=lower_whisker, y1=lower_whisker,
                      line=dict(color=current_color))
        fig.add_shape(type="line",
                      x0=index - 0.4, x1=index + 0.4,
                      y0=upper_whisker, y1=upper_whisker,
                      line=dict(color=current_color))
        
        # Add vertical lines connecting the top and bottom of the box to the custom whiskers
        fig.add_shape(type="line",
                    x0=index, x1=index,
                    y0=upper_limit, y1=upper_whisker,
                    line=dict(color=current_color))
        fig.add_shape(type="line",
                    x0=index, x1=index,
                    y0=lower_limit, y1=lower_whisker,
                    line=dict(color=current_color))

    fig.update_layout(
        title='Box Plot of Data for First 500ms',
        xaxis_title='Columns',
        yaxis_title='Value',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=300*len(subset_df.columns)
    )
    
    if plot_output:
        fig.show()
    
    results_df = pd.DataFrame(results)
    return results_df

def calculate_and_plot_snr(results_df, plot_output=True):
    """
    Calculate the Signal to Noise Ratio (SNR) and optionally create a vertical bee swarm plot with a single column.
    
    Parameters:
    - results_df: DataFrame containing the 2 whiskers and 2 limits for each boxplot.
    - plot_output: Boolean, if True (default), the SNR plot will be produced.
    
    Returns:
    - snr_df: DataFrame containing the SNR for each column.
    """
    results_df['SNR'] = (abs(results_df['Upper Whisker']) + abs(results_df['Lower Whisker'])) / (abs(results_df['Upper Limit']) + abs(results_df['Lower Limit']))
    snr_df = results_df[['Column', 'SNR']]

    if plot_output:
        colors = cycle(go.Figure().full_figure_for_development()['layout']['colorway'])
        fig = go.Figure()

        for index, row in snr_df.iterrows():
            current_color = next(colors)
            fig.add_trace(go.Scatter(
                x=[1] * len(snr_df),  # all points in a single column
                y=[row['SNR']],
                mode='markers+text',
                marker=dict(size=10, color=current_color),
                name=row['Column']
            ))

        fig.update_layout(
            title='Signal to Noise Ratio (SNR) for Each Column',
            xaxis=dict(showticklabels=False, title=''),  # hide x-axis
            yaxis_title='SNR',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            width=500
        )

        fig.update_traces(hoverlabel=dict(namelength=-1))

        fig.show()

    return snr_df

def plot_fft_of_signals(normalized_df, sampling_rate=500):
    """
    Plot the FFT for signals in a dataframe using plotly.graph_objects.
    
    Parameters:
    - normalized_df: DataFrame with time-series signals.
    - sampling_rate: Sampling rate of the signals in Hz.
    """
    # Calculate Nyquist frequency (half of the sampling rate)
    nyquist = 0.5 * sampling_rate
    freqs = np.linspace(0, nyquist, len(normalized_df) // 2)
    
    fig = go.Figure()

    # Plot FFT for each column in the dataframe
    for column in normalized_df.columns:
        # Compute FFT and take the absolute value
        fft_vals = np.fft.rfft(normalized_df[column])
        fft_magnitude = np.abs(fft_vals)
        
        fig.add_trace(go.Scatter(x=freqs, y=fft_magnitude[:len(freqs)], mode='lines', name=column))
        
    # Update layout
    fig.update_layout(
        title="FFT of Signals",
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.show()


    
