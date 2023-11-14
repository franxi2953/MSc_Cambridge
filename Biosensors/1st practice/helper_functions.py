import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import cycle
from scipy.fftpack import fft
from scipy.interpolate import interp1d
import warnings

# Suppress UserWarning from Plotly's Kaleido module
warnings.filterwarnings("ignore", category=UserWarning, module="plotly.io._kaleido")

# Suppress SettingWithCopyWarning from pandas
warnings.filterwarnings("ignore", category=pd.core.common.SettingWithCopyWarning)

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

        df = df.dropna()

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

def plot_raw_and_normalized(df, columns_to_plot=None, plot_output=True):
    """
    Plot raw data and its corresponding baseline-corrected and normalized data.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe containing raw data.
    - columns_to_plot (list or range, optional): A list of column names to plot or a range.
        If None, plots all columns. Default is None.
    - plot_output (bool, optional): Whether to display the plot. Default is True.
    
    Returns:
    - normalized_df (pd.DataFrame): A dataframe containing the baseline-corrected and normalized data.
    
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
    subplot_titles = [name for pair in col_names for name in (pair, f"{pair} (Normalized)")]
    fig = make_subplots(rows=n_columns, cols=2, shared_xaxes=True, subplot_titles=subplot_titles)

    normalized_df = pd.DataFrame(index=df.index)

    # Plotting columns
    for i, col_name in enumerate(col_names):
        current_color = next(colors)
        
        # Raw data
        raw_data = df[col_name]
        fig.add_trace(go.Scattergl(x=df.index, y=raw_data, mode='lines', name=col_name, line=dict(color=current_color)), row=i+1, col=1)

        # Baseline correction
        baseline = raw_data.iloc[:500].mean()
        corrected_data = raw_data - baseline
        
        # Normalization
        normalized_data = corrected_data #min_max_normalize(corrected_data)
        normalized_df[col_name + '_normalized'] = normalized_data
        
        # Plotting normalized data
        fig.add_trace(go.Scattergl(x=df.index, y=normalized_data, mode='lines', name=f"{col_name} (Normalized)", line=dict(color=current_color)), row=i+1, col=2)

    # Update layout
    fig.update_layout(title='', xaxis_title='Milliseconds', plot_bgcolor='white', paper_bgcolor='white', width=1400, height=300*n_columns, showlegend=False)
    
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

def extract_envelope(signal_df, column_name, window_size=50, plot_output=True):
    """
    Extracts the envelope of an EMG signal using a sliding window method and optionally plots the result.

    :param signal_df: A pandas DataFrame containing the EMG data.
    :param column_name: The name of the column in the DataFrame that contains the EMG signal.
    :param window_size: The size of the sliding window for envelope extraction.
    :param plot_output: If True, the function will plot the original signal and the envelope.
    :return: A pandas DataFrame with the original signal and the extracted envelope.
    """
    # Ensure the signal column exists in the DataFrame
    if column_name not in signal_df.columns:
        raise ValueError(f"The specified column '{column_name}' does not exist in the DataFrame.")

    # Extract the signal
    signal = signal_df[column_name].values

    # Initialize the envelope arrays
    positive_envelope = np.zeros_like(signal)
    negative_envelope = np.zeros_like(signal)

    # Perform envelope extraction using a sliding window
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2)
        window = signal[start:end]
        
        positive_envelope[i] = np.max(window[window >= 0]) if np.any(window >= 0) else 0
        negative_envelope[i] = np.min(window[window < 0]) if np.any(window < 0) else 0

    # Add the envelope to the DataFrame
    envelope_column_name = f"{column_name}_envelope"
    signal_df[envelope_column_name + '_positive'] = positive_envelope
    signal_df[envelope_column_name + '_negative'] = negative_envelope

    # Optionally create the plot
    if plot_output:
        fig = go.Figure()

        # Add the original signal trace
        fig.add_trace(go.Scatter(x=signal_df.index, y=signal, mode='lines', name='Original Signal'))

        # Add the positive envelope trace
        fig.add_trace(go.Scatter(x=signal_df.index, y=positive_envelope, mode='lines', name='Envelope (Positive)'))

        # Add the negative envelope trace
        fig.add_trace(go.Scatter(x=signal_df.index, y=negative_envelope, mode='lines', name='Envelope (Negative)'))

        # Update the layout
        fig.update_layout(
            title='EMG Signal and Envelope',
            xaxis_title='Time (ms)',
            yaxis_title='Amplitude',
            plot_bgcolor='white',
            showlegend=True
        )

        # Show the plot
        fig.show()

    return signal_df

def calculate_derivative(signal):
    """
    Calculate the derivative of a signal.
    """
    return np.diff(signal, prepend=signal[0])

def rough_segmentation(signal_df, envelope_column_name, lambda_param=0.1, min_segment_distance=80):
    """
    Performs rough segmentation of an EMG signal based on its envelope and the derivative of the envelope.

    :param signal_df: A pandas DataFrame containing the EMG data and envelope.
    :param envelope_column_name: The name of the column in the DataFrame that contains the envelope.
    :param lambda_param: The empirical parameter λ used to calculate the threshold.
    :param min_segment_distance: The minimum distance between segment openings and closings.
    :return: A pandas DataFrame with additional columns indicating the start and end of active segments.
    """
    # Drop rows with NaN values in the specified envelope column
    signal_df = signal_df.dropna(subset=[envelope_column_name])

    # Ensure the envelope column exists in the DataFrame
    if envelope_column_name not in signal_df.columns:
        raise ValueError(f"The specified envelope column '{envelope_column_name}' does not exist in the DataFrame.")

    # Extract the envelope
    envelope = signal_df[envelope_column_name].values

    # Calculate the derivative of the envelope
    derivative = calculate_derivative(envelope)

    # Calculate the threshold
    threshold = lambda_param * np.max(envelope)

    # Initialize the segments array
    segments = np.zeros_like(envelope, dtype=int)

    # State variable to track whether we are inside a segment
    inside_segment = False

    # Perform rough segmentation
    for i in range(1, len(envelope)):
        if envelope[i] > threshold and derivative[i] > 0 and not inside_segment:
            segments[i] = 1  # Start of a segment
            inside_segment = True
        elif envelope[i] < threshold and derivative[i] < 0 and inside_segment:
            segments[i] = -1  # End of a segment
            inside_segment = False

    # Add the segments and derivative to the DataFrame
    segments_column_name = f'{envelope_column_name}_segments'
    derivative_column_name = f'{envelope_column_name}_derivative'
    
    signal_df[segments_column_name] = segments
    signal_df[derivative_column_name] = derivative

    # Remove close segments
    signal_df = remove_close_segments(signal_df, segments_column_name, min_segment_distance)

    return signal_df

def remove_close_segments(signal_df, segments_column_name, min_segment_distance=80):
    """
    Removes segment openings and closings that are too close to each other.

    :param signal_df: A pandas DataFrame containing the segments information.
    :param segments_column_name: The name of the column in the DataFrame that contains the segment indicators.
    :param min_segment_distance: The minimum distance between segment openings and closings.
    :return: A pandas DataFrame with updated segment indicators.
    """
    segments = signal_df[segments_column_name].values
    i = 0
    while i < len(segments):
        if segments[i] in [-1, 1]:
            j = i + 1
            while j < len(segments) and segments[j] not in [-1, 1]:
                j += 1
            if j < len(segments) and j - i < min_segment_distance:
                segments[i] = 0
                segments[j] = 0
            i = j
        else:
            i += 1
    signal_df[segments_column_name] = segments
    return signal_df

def adjust_segments_based_on_integral(signal_df, envelope_column_name, segments_column_name, envelope_window_size):
    """
    Adjusts the segmentation of a signal based on the integral of the signal.

    :param signal_df: A pandas DataFrame containing the EMG data, envelope, and segments.
    :param envelope_column_name: The name of the column in the DataFrame that contains the envelope.
    :param segments_column_name: The name of the column in the DataFrame that contains the segment indicators.
    :param envelope_window_size: The size of the window used for envelope extraction.
    :return: A pandas DataFrame with the adjusted segments.
    """
    # Calculate the Integral Electromyography (IEMG)
    dt = envelope_window_size / 2  # ∆t is half of the envelope window
    iemg = np.cumsum(np.abs(signal_df[envelope_column_name].values)) * dt

    # Adjust the start and end points of each segment
    segments = signal_df[segments_column_name].values
    adjusted_segments = np.copy(segments)
    start_idx = None
    for i in range(1, len(segments)):
        if segments[i] == 1:
            start_idx = i
        elif segments[i] == -1 and start_idx is not None:
            # Calculate the envelope length L
            L = i - start_idx
            
            # Define the range of the moving window as one tenth of envelope length L
            window_range = L // 10
            
            # Adjust the start point
            new_start_idx = start_idx - window_range + np.argmax(iemg[start_idx - window_range:start_idx + window_range])
            adjusted_segments[start_idx] = 0
            adjusted_segments[new_start_idx] = 1
            
            # Adjust the end point
            new_end_idx = i - window_range + np.argmax(iemg[i - window_range:i + window_range])
            adjusted_segments[i] = 0
            adjusted_segments[new_end_idx] = -1
            
            start_idx = None

    signal_df[segments_column_name] = adjusted_segments
    return signal_df

def calculate_segments(signal_df, column_name, window_size=50, lambda_param=0.1, min_segment_distance=80, plot_output=True, plot_adjustment=False, plot_envelope=False):
    """
    Calculates the envelope of a signal, performs rough segmentation, adjusts the segmentation, and optionally plots the results.

    :param signal_df: A pandas DataFrame containing the EMG data.
    :param column_name: The name of the column in the DataFrame that contains the EMG signal.
    :param window_size: The size of the sliding window for envelope extraction.
    :param lambda_param: The empirical parameter λ used to calculate the threshold for segmentation.
    :param min_segment_distance: The minimum distance between segment openings and closings.
    :param plot_output: If True, the function will plot the original signal, envelope, and segments.
    :param plot_adjustment: If True, the function will plot the segments before and after adjustment.
    :return: A pandas DataFrame with the original signal, envelope, and segments.
    """
    # Step 1: Extract Envelope
    result_df = extract_envelope(signal_df, column_name, window_size, plot_output=plot_envelope)
    
    # Step 2: Rough Segmentation
    envelope_column_name = f"{column_name}_envelope"
    positive_envelope_column_name = f"{envelope_column_name}_positive"
    negative_envelope_column_name = f"{envelope_column_name}_negative"
    segments_column_name = f"{positive_envelope_column_name}_segments"
    segmented_df = rough_segmentation(result_df, positive_envelope_column_name, lambda_param, min_segment_distance)

    # Step 3: Segmentation Adjustment
    if plot_adjustment and plot_output:
        # Plot before adjustment
        plot_segments(segmented_df, column_name, positive_envelope_column_name, negative_envelope_column_name, segments_column_name, title='Segments Before Adjustment')
    
    # Adjust and merge segments
    adjusted_df = adjust_segments(segmented_df, segments_column_name)
    
    if plot_output:
        # Plot after adjustment (or the only plot if plot_adjustment is False)
        plot_segments(adjusted_df, column_name, positive_envelope_column_name, negative_envelope_column_name, segments_column_name, title='Segments After Adjustment')

    return adjusted_df

def adjust_segments(signal_df, segments_column, adjustment=100):
    """
    Adjusts the segments in the signal DataFrame.

    :param signal_df: A pandas DataFrame containing the EMG data and segments.
    :param segments_column: The name of the column in the DataFrame that contains the segment indicators.
    :param adjustment: The number of indices to adjust the start and end of each segment. Default is 10.
    :return: A pandas DataFrame with the adjusted segments.
    """
    adjusted_segments = np.copy(signal_df[segments_column].values)
    original_segments = np.copy(adjusted_segments)

    for i in range(1, len(adjusted_segments) - 1):
        if original_segments[i - 1] != 1 and original_segments[i] == 1:
            start = max(i - adjustment, 0)
            adjusted_segments[start] = 1  # Set new start of segment
            adjusted_segments[start+1:i] = 0  # Set values between new start and original start to 0
            adjusted_segments[i] = 0  # Set original start back to 0
        elif original_segments[i] == -1 and original_segments[i + 1] != -1:
            end = min(i + adjustment, len(adjusted_segments) - 1)
            adjusted_segments[i] = 0  # Set original end back to 0
            adjusted_segments[i+1:end] = 0  # Set values between original end and new end to 0
            adjusted_segments[end] = -1  # Set new end of segment

    signal_df[segments_column] = adjusted_segments
    return signal_df

def plot_segments(df, signal_column, positive_envelope_column, negative_envelope_column, segments_column, title='EMG Signal, Envelope, and Segments'):
    """
    Plots the EMG signal, its positive and negative envelopes, and the segments.

    :param df: A pandas DataFrame containing the EMG data, envelopes, and segments.
    :param signal_column: The name of the column in the DataFrame that contains the EMG signal.
    :param positive_envelope_column: The name of the column in the DataFrame that contains the positive envelope.
    :param negative_envelope_column: The name of the column in the DataFrame that contains the negative envelope.
    :param segments_column: The name of the column in the DataFrame that contains the segment indicators.
    :param title: The title of the plot.
    """
    fig = go.Figure()

    # Add the original signal trace
    fig.add_trace(go.Scatter(x=df.index, y=df[signal_column], mode='lines', name='Original Signal'))

    # Add the positive envelope trace
    fig.add_trace(go.Scatter(x=df.index, y=df[positive_envelope_column], mode='lines', name='Positive Envelope'))

    # Add the negative envelope trace
    fig.add_trace(go.Scatter(x=df.index, y=df[negative_envelope_column], mode='lines', name='Negative Envelope'))

    # Add vertical lines for segment starts and ends
    for idx, segment in zip(df.index, df[segments_column]):
        if segment == 1:
            fig.add_shape(go.layout.Shape(type="line", x0=idx, x1=idx, y0=0, y1=1, yref="paper", line=dict(color="Green", width=4)))
        elif segment == -1:
            fig.add_shape(go.layout.Shape(type="line", x0=idx, x1=idx, y0=0, y1=1, yref="paper", line=dict(color="Red", width=4)))



    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title='Time (ms)',
        yaxis_title='Amplitude',
        plot_bgcolor='white',
        showlegend=True
    )

    # Show the plot
    fig.show()

def plot_segments_overlay(df, signal_column, segments_column, plot_output=False, transparency=0.2, full_wave_rectification=False, rms_window_size=10):
    """
    Plots all segments of a signal overlaid on top of each other with specified transparency, time-normalized.
    Optionally performs full-wave rectification on the signal and calculates the RMS average of all the segments.

    :param df: A pandas DataFrame containing the EMG data and segments.
    :param signal_column: The name of the column in the DataFrame that contains the EMG signal.
    :param segments_column: The name of the column in the DataFrame that contains the segment indicators.
    :param transparency: The transparency of the overlaid segments. Default is 0.2.
    :param full_wave_rectification: If True, performs full-wave rectification on the signal. Default is False.
    :param rms_window_size: The window size for RMS calculation in samples. Default is 10.
    :return: A pandas Series representing the RMS average of all the segments, if full_wave_rectification is True. Otherwise, None.
    """
    fig = go.Figure()

    # Find start and end indices of all segments
    starts = np.where(df[segments_column].values == 1)[0]
    ends = np.where(df[segments_column].values == -1)[0]

    # Check if the number of starts and ends match
    if len(starts) != len(ends):
        print("Warning: The number of segment starts and ends do not match.")
        return

    # Find the maximum time span of all segments
    max_time_span = max(ends - starts)

    # Store all normalized segments
    normalized_segments = []

    # Plot each segment, time-normalized
    for start, end in zip(starts, ends):
        segment = df[signal_column].iloc[start:end+1]
        if full_wave_rectification:
            segment = segment.abs()
        time_normalized = np.linspace(0, max_time_span, len(segment))
        normalized_segments.append(np.interp(np.arange(max_time_span+1), time_normalized, segment))
        fig.add_trace(go.Scatter(x=np.arange(max_time_span+1), y=normalized_segments[-1], mode='lines', line=dict(width=2, color='#636EFA'), opacity=transparency))

    # Calculate the RMS average of all segments if full_wave_rectification is True
    average_segment = None
    if full_wave_rectification:
        squared_segments = np.square(normalized_segments)
        rms_segments = np.sqrt(pd.DataFrame(squared_segments).rolling(window=rms_window_size, min_periods=1, axis=1).mean())
        average_segment = rms_segments.mean().values

        # Add the average segment trace
        fig.add_trace(go.Scatter(x=np.arange(max_time_span+1), y=average_segment, mode='lines', line=dict(width=3, color='red'), name='RMS Average Segment'))

    # Update the layout
    fig.update_layout(
        title='Overlay of Time-Normalized Segmented EMG Signals',
        xaxis_title='Time (normalized)',
        yaxis_title='Amplitude',
        plot_bgcolor='white',
        showlegend=True,
        width=600
    )

    # Show the plot
    if plot_output:
        fig.show()

    return pd.Series(average_segment) if average_segment is not None else None

def calculate_snr(signal, noise_duration=100):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of a signal.

    :param signal: The signal time series (numpy array or pandas Series).
    :param noise_duration: Duration of noise at the beginning and end of the signal in milliseconds.
    :return: The SNR in decibels (dB).
    """
    # Assuming the signal is sampled at 1000 Hz (1 ms per sample)
    sample_rate = 1000
    noise_samples = int(noise_duration * sample_rate / 1000)

    # Extract noise segments from the beginning and end of the signal
    noise = np.concatenate([signal[:noise_samples], signal[-noise_samples:]])

    # Calculate the power of the signal
    signal_power = np.mean(np.square(signal))

    # Calculate the power of the noise
    noise_power = np.mean(np.square(noise))

    # Calculate the SNR
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_and_plot_fft(normalized_df, columns_to_plot=None, plot_output=True, apply_filter=False):
    """
    Compute and plot the FFT of specified columns from the input DataFrame.
    
    Parameters:
    - normalized_df (pd.DataFrame): The input dataframe containing normalized data.
    - columns_to_plot (list or range, optional): A list of column names to plot or a range.
        If None, plots all columns. Default is None.
    - plot_output (bool, optional): Whether to display the plot. Default is True.
    
    Returns:
    - fft_df (pd.DataFrame): A dataframe containing the FFT results.
    """
    
    # Selecting columns to plot
    if columns_to_plot is None:
        col_names = normalized_df.columns.tolist()
    elif isinstance(columns_to_plot, range):
        col_names = normalized_df.columns[columns_to_plot].tolist()
    else:
        col_names = columns_to_plot

    n_columns = len(col_names)
    colors = cycle(go.Figure().full_figure_for_development()['layout']['colorway'])
    subplot_titles = [f"{name} (FFT)" for name in col_names]
    fig = make_subplots(rows=n_columns, cols=1, shared_xaxes=True, subplot_titles=subplot_titles)
    
    # Generating the frequency bins
    freqs = np.fft.fftfreq(len(normalized_df.index), d=2e-3)  # d is the inverse of the sampling rate
    
    # Filtering out negative frequencies
    positive_freqs = freqs[freqs >= 0]
    
    fft_df = pd.DataFrame(index=positive_freqs)

    # Processing and plotting columns
    for i, col_name in enumerate(col_names):
        current_color = next(colors)
        
        # Handling NaN values by filling with zeros
        data_no_nan = normalized_df[col_name].fillna(0)
        
        # Running the FFT
        fft_result = np.fft.fft(data_no_nan)
        
        # Apply filtering if the flag is set
        if apply_filter:
            fft_result = filtering(fft_result, freqs)

        # Filtering out negative frequencies from the FFT result
        positive_fft_result = fft_result[freqs >= 0]
        
        fft_df[col_name + '_fft'] = positive_fft_result
        
        # Plotting FFT result
        fig.add_trace(go.Scattergl(x=positive_freqs, y=np.abs(positive_fft_result), mode='lines', name=f"{col_name} (FFT)", line=dict(color=current_color)), row=i+1, col=1)

    # Update layout with your specified width of 600
    fig.update_layout(title='FFT Analysis', xaxis_title='Frequency (Hz)', plot_bgcolor='white', paper_bgcolor='white', width=600, height=500*n_columns, showlegend=False)
    
    if plot_output:
        fig.show()
    
    return fft_df

def filtering(fft_result, freqs):
    """
    Filters out frequencies between 0 and 10, and between 49 and 51.
    
    Parameters:
    - fft_result (np.array): The FFT result.
    - freqs (np.array): The corresponding frequencies.
    
    Returns:
    - filtered_fft (np.array): The filtered FFT result.
    """
    # Create masks for the frequencies to be filtered out
    mask = ((freqs >= 10) & (freqs <= 49)) | ((freqs >= 52))
    
    # Apply the masks to filter out the undesired frequencies
    filtered_fft = fft_result * mask
    
    return filtered_fft

def plot_fft_overlay(df1, df2, opacity=0.2):
    # Create figure
    fig = go.Figure()

    # Extract data from the first dataframe
    x1 = df1.index
    y1 = df1.iloc[:, 0].apply(np.abs)  # Selects the first column and computes the magnitude
    
    # Add trace for the first dataframe
    fig.add_trace(go.Scattergl(x=x1, y=y1, mode='lines', name='DataFrame 1',opacity=1))
    
    # Extract data from the second dataframe
    x2 = df2.index
    y2 = df2.iloc[:, 0].apply(np.abs)  # Selects the first column and computes the magnitude
    
    # Add trace for the second dataframe
    fig.add_trace(go.Scattergl(x=x2, y=y2, mode='lines', name='DataFrame 2', opacity=opacity))
    
    # Update layout for better readability
    fig.update_layout(title='FFT Overlay Plot',
                      xaxis_title='Frequency (Hz)',
                      yaxis_title='Magnitude',
                      plot_bgcolor='white',
                      paper_bgcolor='white',
                      width=1000)
    
    # Show plot
    fig.show()

def compute_signal_rms(df, signal_column=None, full_wave_rectification=False, rms_window_size=10, plot_output=False):

    """
    Computes the RMS of a signal with optional full-wave rectification and plotting.

    :param df: A pandas DataFrame containing the EMG data.
    :param signal_column: The name of the column in the DataFrame that contains the EMG signal.
    :param full_wave_rectification: If True, performs full-wave rectification on the signal. Default is False.
    :param rms_window_size: The window size for RMS calculation in samples. Default is 10.
    :param plot_output: If True, plots the signal and its RMS using Plotly. Default is False.
    :return: A pandas Series representing the RMS of the signal.
    """

    df = df.dropna()

    if signal_column is None:
        signal_column = df.columns[0]
    
    signal = df[signal_column]
    if full_wave_rectification:
        signal = signal.abs()
    
    squared_signal = np.square(signal)
    rms_signal = np.sqrt(squared_signal.rolling(window=rms_window_size, min_periods=1).mean())
    
    if plot_output:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=signal.abs(), mode='lines', name='Original Signal'))
        fig.add_trace(go.Scatter(x=df.index, y=rms_signal, mode='lines', name='RMS Signal'))
        fig.update_layout(
            title='Original and RMS Signal',
            xaxis_title='Time',
            yaxis_title='Amplitude',
            plot_bgcolor='white',
            showlegend=False,
            width=1000
        )
        fig.show()
    
    return rms_signal

def plot_mapping(a=1, b=0):
    # Generate data
    delta_S = np.linspace(-10, 10, 400)
    motor_movement = 1 / (1 + np.exp(-(a * (delta_S - b))))

    # Create plot
    fig = go.Figure(data=go.Scatter(x=delta_S, y=motor_movement, mode='lines'))

    # Set plot title and labels
    fig.update_layout(
        title="Mapping of ΔS to Motor Movement",
        xaxis_title="ΔS (EMG signal delta from baseline)",
        yaxis_title="Motor Movement",
        template="plotly",
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        width = 1000
    )

    # Show plot
    fig.show()