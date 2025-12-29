"""
Helper functions for reinforcement learning calibration tool
"""

import pandas as pd
import numpy as np
import subprocess
from datetime import datetime
import time

# Read in any DailyResults file and return dataframe of calibration data indexed by date
def results_interpreter(csv_path, calibration_data):
    results_df = pd.read_csv(csv_path, usecols=['Year', 'Day', calibration_data])
    start_year = results_df['Year'].iloc[0]
    start_date = f'1/1/{start_year}'
    date_range = pd.date_range(start=start_date, periods=len(results_df), freq='D')
    results_df['Date'] = date_range
    results_df.set_index('Date', inplace=True)
    results_df = results_df.drop(columns=['Year', 'Day'])
    return results_df

# Align dataframes and return data grouped by year
def align_group_data(observed_df, results_df):
    aligned = results_df.join(observed_df, how='inner', lsuffix='_sim', rsuffix='_obs')
    intersection = results_df.index.intersection(observed_df.index)
    aligned = aligned.dropna()
    grouped = aligned.groupby(aligned.index.year)
    for year, data in grouped:
        if len(data) < 315:
            print(f'WARNING: observed data in year {year} is missing 50+ values.')
    return grouped


# Functions to calculate metrics of grouped data and return as dictionary by year
def calculate_nse(grouped_data, calibration_data):
    nse_by_year = {}
    for year, data in grouped_data:
        obs = data[f'{calibration_data}_obs']
        sim = data[f'{calibration_data}_sim']
        numerator = np.sum((obs-sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        nse = 1 - (numerator / denominator)
        if nse < -2:  # Set a limit to NSE for better GPR fitting
            nse = -2
        nse_by_year[year] = nse
    return nse_by_year

def calculate_r2(grouped_data, calibration_data):
    r2_by_year = {}
    for year, data in grouped_data:
        obs = data[f'{calibration_data}_obs']
        sim = data[f'{calibration_data}_sim']
        obs_mean = np.mean(obs)
        sim_mean = np.mean(sim)
        numerator = np.sum((obs - obs_mean) * (sim - sim_mean)) ** 2
        denominator = np.sum((obs - obs_mean) ** 2) * np.sum((sim - sim_mean) ** 2)
        r2_by_year[year] = numerator / denominator
    return r2_by_year

def calculate_soar_summer(grouped_data, calibration_data):
    soar_summer_by_year = {}
    for year, data in grouped_data:
        obs = data[f'{calibration_data}_obs'][data.index.month.isin([6, 7, 8, 9])]  # Extra filter for June through September
        sim = data[f'{calibration_data}_sim'][data.index.month.isin([6, 7, 8, 9])]
        obs_sum = np.sum(obs)
        sim_sum = np.sum(sim)
        soar_summer_by_year[year] = sim_sum / obs_sum
        if soar_summer_by_year[year] > 1.0:
            soar_summer_by_year[year] = 1.0 - abs(1.0 - soar_summer_by_year[year])
    return soar_summer_by_year

def calculate_soar(grouped_data, calibration_data):
    soar_by_year = {}
    for year, data in grouped_data:
        obs = data[f'{calibration_data}_obs']
        sim = data[f'{calibration_data}_sim']
        obs_sum = np.sum(obs)
        sim_sum = np.sum(sim)
        soar_by_year[year] = sim_sum / obs_sum
        if soar_by_year[year] > 1.0:
            soar_by_year[year] = 1.0 - abs(1.0 - soar_by_year[year])
    return soar_by_year


# Function to calculate APES score
def calculate_apes(years, results, weights):
    apes_score_list = []
    if isinstance(years, int):
        years = [years]
    for year in years:
        partial_apes = []
        for dict, weight in zip(results, weights):
            if year not in dict:
                continue  # Allows years with missing data
            metric = dict[year]
            partial_apes.append(metric*weight)
        if not partial_apes:
            continue
        apes_score = sum(partial_apes) / len(partial_apes)
        apes_score_list.append(apes_score)
    apes_score = np.mean(apes_score_list)
    return apes_score

# Function to evaluate model by the difference in APES score
def evaluate_model(eval_year, results, default_results, weights):
    default_apes = calculate_apes(eval_year, default_results, weights)
    result_apes = calculate_apes(eval_year, results, weights)
    reward = result_apes - default_apes
    return reward

# Function to build VELMA command and run as a sub-process
def run_velma(parallel_flag, allocated_memory, jar_path, xml_path, start_year, end_year, end_data, parameter_modifiers, start_data=None, max_processes=1):
    if parallel_flag:
        command = ["java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaParallelCmdLine", xml_path,
                  f"--maxProcesses={max_processes}",
                  f'--kv="/calibration/VelmaInputs.properties/syear",{start_year}',
                  f'--kv="/calibration/VelmaInputs.properties/eyear",{end_year}',
                  f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName",{end_data}',
                  *parameter_modifiers]
        if start_data:
            command.append(f'--kv="/startups/VelmaStartups.properties/setStartStateSpatialDataLocationName",{start_data}')
        command_str = ' '.join(command)
        try:
            subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except:
            raise RuntimeError(f"VELMA failed using command: {command_str}")
    else:
        command = ["java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaSimulatorCmdLine", xml_path,
                   f'--kv="/calibration/VelmaInputs.properties/syear",{start_year}',
                   f'--kv="/calibration/VelmaInputs.properties/eyear",{end_year}',
                   f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName",{end_data}',
                   *parameter_modifiers]
        if start_data:
            command.append(f'--kv="/startups/VelmaStartups.properties/setStartStateSpatialDataLocationName",{start_data}')
        command_str = ' '.join(command)
        try:
            subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except:
            raise RuntimeError(f"VELMA failed using command: {command_str}")

# Function to find outlet_id
def row_col_to_index(asc_file_path, row, col):
    with open(asc_file_path, "r") as f:
        for line in f:
            if line.lower().startswith("ncols"):
                ncols = int(line.split()[1])
                break
    return row * ncols + col

# Logging helper functions
def log_message(msg, logfile):
    with open(logfile, "a") as f:
        f.write(msg + "\n")
        f.flush()

def log_with_timestamp(message, logfile):
    log_message(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", logfile)

def start_timer():
    return time.time()

def elapsed_time(start):
    return round(time.time() - start, 0)  # seconds


# Two LHS sampling functions
def lhs(n_samples, n_params):
    result = np.zeros((n_samples, n_params))

    # For each parameter dimension
    for j in range(n_params):
        # Cut [0,1] into n_samples
        cut = np.linspace(0, 1, n_samples + 1)

        # Random points inside each interval
        u = np.random.rand(n_samples)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])

        # Shuffle so each sample gets exactly one stratum
        np.random.shuffle(points)

        # Store
        result[:, j] = points

    return result

def generate_lhs_param_sets(velma_parameters, n_samples):
    param_names = list(velma_parameters.keys())
    n_params = len(param_names)

    # LHS in [0,1]
    unit_samples = lhs(n_samples, n_params)

    # Scale to parameter ranges
    scaled = np.zeros_like(unit_samples)
    for i, p in enumerate(param_names):
        pmin = velma_parameters[p]['min']
        pmax = velma_parameters[p]['max']
        scaled[:, i] = pmin + unit_samples[:, i] * (pmax - pmin)
        scaled = np.round(scaled, 4)

    return param_names, scaled
