"""
Helper functions for reinforcement learning calibration tool
"""

import pandas as pd
import numpy as np
import subprocess

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
def align_group_data(observed_df, results_df, calibration_data):
    aligned = results_df.join(observed_df, lsuffix='_sim', rsuffix='_obs')
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
        nse_by_year[year] = 1 - (numerator / denominator)
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
def calculate_apes(year, results, weights):
    partial_apes = []
    for dict, weight in zip(results, weights):
        metric = dict[year]
        partial_apes.append(metric*weight)
    apes_score = sum(partial_apes) / len(partial_apes)
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
