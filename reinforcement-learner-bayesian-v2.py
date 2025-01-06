import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import shutil
from skopt import gp_minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import random
from sklearn.inspection import permutation_importance


# Define the required run parameters and directories
start_year = 1987  # Start the model spin-up from this year
start_obs_data_year = 1981  # Enter the year the observed data starts from (be sure to check the observed data file)
start_learning_year = 1991
end_year = 2021
allocated_memory = "-Xmx5G"
jar_path = "C:/Users/thorn/OneDrive/Desktop/JVelma_dev-test_v003.jar"
working_directory = 'C:/Users/thorn/Documents/VELMA_Watersheds/Huge'
xml_name = 'WA_Huge30m_19Dec2024'
xml_path = f'{working_directory}/XML/{xml_name}.xml'
results_folder_root = f'{working_directory}/Results'
q_table_output = f'{results_folder_root}/q-table.csv'
running_average_output = f'{results_folder_root}/running-averages.csv'
figure_path = f'{results_folder_root}/Figures'
velma_parallel = True
epsilon = 0.2 # set the rate of random exploration here
default_results = f'{results_folder_root}/MULTI_WA_Huge30m_19Dec2024_default/Results_75524/DailyResults.csv'  # must be DailyResults.csv
calibration_data = 'Runoff_All(mm/day)_Delineated_Average'  # must EXACTLY match a column in the DailyResults file
observed_file = f'{working_directory}/Data_Inputs30m/m_7_Observed/USGS12073500_Huge_streamflow_1981_2021.csv'
metric = 'nse'  # Valid options are 'nse' or 'r2'
# Required run parameters if velma_parallel is True
outlet_id = '75524'
max_processes = "6"

# Define starting and allowable VELMA parameter ranges
# 'value' must match value used in default results
velma_parameters = {
    '/calibration/VelmaCalibration.properties/setGroundwaterStorageFraction': {'name': 'GroundwaterStorageFraction','value': 0, 'min': 0.0, 'max': 0.15},
    '/calibration/VelmaCalibration.properties/f_ksv': {'name': 'VerticalKs', 'value': 0.0013, 'min': 0.001, 'max': 0.002},
    '/calibration/VelmaCalibration.properties/f_ksl': {'name': 'LateralKs', 'value': 0.00155, 'min': 0.001, 'max': 0.002},
    # '/soil/Medium_CN24/soilColumnDepth': {'name': 'MediumSoilDepth','value': 1080, 'min': 1080, 'max': 1300},
    # '/soil/Shallow_CN24/soilColumnDepth': {'name': 'ShallowSoilDepth','value': 470, 'min': 470, 'max': 1080},
    # '/soil/Deep_CN24/soilColumnDepth': {'name': 'DeepSoilDepth','value': 1530, 'min': 1300, 'max': 3000},
    # '/soil/Medium_CN24/surfaceKs': {'name': 'MediumKs','value': 891, 'min': 400, 'max': 2000},
    # '/soil/Shallow_CN24/surfaceKs': {'name': 'ShallowKs','value': 1200, 'min': 400, 'max': 2000},
    # '/soil/Deep_CN24/surfaceKs': {'name': 'DeepKs','value': 1200, 'min': 400, 'max': 2000}
    # Can add/remove parameters, but doing so necessitates manual changes to q-table and running-average table
}

# Basic checks on the input directories
for directory in [jar_path, working_directory, results_folder_root, xml_path, default_results, observed_file]:
    if not os.path.exists(directory):
        print(f'FATAL WARNING: {directory} does not exist. Killing program. \nCheck directories and restart script.')
        exit()
if not os.path.exists(figure_path):
    os.mkdir(figure_path)

# Outdated file / folder cleaning
if os.path.exists(f'{results_folder_root}/{xml_name}'):
    print(f'FATAL WARNING: {results_folder_root}/{xml_name} already exists. Killing program. \nEither delete or rename this folder and then restart the script.')
    exit()
print('Input directories are valid.')
for year in range(start_learning_year, end_year+1):
    if os.path.exists(f'{results_folder_root}/Results_{year}'):
        print(f'Removing outdated results folder Results_{year}')
        shutil.rmtree(f'{results_folder_root}/Results_{year}')

# Read in the observed data and assign an index by date
observed_df = pd.read_csv(observed_file, usecols=[0], header=None, names=[calibration_data])
start_date = f'1/1/{start_obs_data_year}'
date_range = pd.date_range(start=start_date, periods=len(observed_df), freq='D')
observed_df['Date'] = date_range
observed_df.set_index('Date', inplace=True)

# Read in any DailyResults file and return dataframe of calibration data indexed by date
def results_interpreter(csv_path):
    results_df = pd.read_csv(csv_path, usecols=['Year', 'Day', calibration_data])
    start_year = results_df['Year'].iloc[0]
    start_date = f'1/1/{start_year}'
    date_range = pd.date_range(start=start_date, periods=len(results_df), freq='D')
    results_df['Date'] = date_range
    results_df.set_index('Date', inplace=True)
    results_df = results_df.drop(columns=['Year', 'Day'])
    return results_df

# Ensure both dataframes are aligned and return data grouped by year
def align_group_data(observed_df, results_df):
    aligned = results_df.join(observed_df, lsuffix='_sim', rsuffix='_obs')
    aligned = aligned.dropna()
    grouped = aligned.groupby(aligned.index.year)
    for year, data in grouped:
        if len(data) < 315:
            print(f'WARNING: observed data in year {year} is missing 50+ values.')
    return grouped

# Functions to calculate NSE and R^2 of any grouped data and return as dictionary by year
def calculate_nse(grouped_data):
    nse_by_year = {}
    for year, data in grouped_data:
        obs = data[f'{calibration_data}_obs']
        sim = data[f'{calibration_data}_sim']
        numerator = np.sum((obs-sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        nse_by_year[year] = 1 - (numerator / denominator)
    return nse_by_year

def calculate_r2(grouped_data):
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

# Compare observed data with baseline results and create a dataframe that contains NSE or R^2 by year
default_df = results_interpreter(default_results)
default_data = align_group_data(observed_df, default_df)
if metric == 'nse':
    default_dict = calculate_nse(default_data)
elif metric == 'r2':
    default_dict = calculate_r2(default_data)
else:
    print(f'FATAL WARNING: {metric} is not valid. Killing program. \nEnter "nse" or "r2".')
    exit()

default_df = pd.DataFrame.from_dict(default_dict, orient='index', columns=[metric])
default_df.index.name = 'Year'
output_file = f'default_{metric}.csv'
default_df.to_csv(f'{results_folder_root}/{output_file}')
print(f'Successfully wrote default {metric} to file {results_folder_root}/{output_file}.')

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

parameter_names = [velma_parameters[param]["name"] for param in velma_parameters]

# List of parameter values for easy dataframe writing
parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]

# Format parameters so they can modify the VELMA run
extended_velma_parameters = velma_parameters.copy()
# This section is only for modification of VELMA run parameters using extended soil types with C/N ratios
soil_type_dict = {
    'CN24': ['CN12', 'CN17'],
}
for soil_type, ratios in soil_type_dict.items():
    for parameter in velma_parameters.keys():
        if soil_type in parameter:
            for ratio in ratios:
                extended_velma_parameters[parameter.replace(soil_type, ratio)] = extended_velma_parameters[parameter]

parameter_modifiers = [f'--kv="{param}",{extended_velma_parameters[param]["value"]}' for param in extended_velma_parameters]

# Run VELMA for spin-up years and output data at end to a folder
end_spinup_year = start_learning_year - 1
end_data = f'{results_folder_root}/Results_{str(end_spinup_year)}'
if os.path.exists(end_data):
    print(f"Accessing data from previous spin-up years.")
else:
    print(f"Running spin-up years {start_year} to {end_spinup_year}.")
    run_velma(velma_parallel, allocated_memory, jar_path, xml_path, start_year, end_spinup_year, end_data, parameter_modifiers, start_data=None, max_processes=max_processes)
    print('Spin-up years complete.')

# Check for spinup folder and rename if necessary
if velma_parallel:
    if not os.path.exists(f'{results_folder_root}/MULTI_{xml_name}_spinup'):
        os.rename(f'{results_folder_root}/MULTI_{xml_name}', f'{results_folder_root}/MULTI_{xml_name}_spinup')
else:
    if not os.path.exists(f'{results_folder_root}/{xml_name}_spinup'):
        os.rename(f'{results_folder_root}/{xml_name}', f'{results_folder_root}/{xml_name}_spinup')
if velma_parallel:
    results_path = f'{results_folder_root}/MULTI_{xml_name}_spinup/Results_{outlet_id}/DailyResults.csv'
else:
    results_path = f'{results_folder_root}/{xml_name}_spinup/DailyResults.csv'

# Calculate metric by year for the simulated results
results_df = results_interpreter(results_path)
grouped = align_group_data(observed_df, results_df)
if metric == 'nse':
    results_dict = calculate_nse(grouped)
elif metric == 'r2':
    results_dict = calculate_r2(grouped)
print(f'End spin-up year had NSE = {results_dict[end_spinup_year]}.')

def evaluate_model(eval_year, metric_dict=results_dict, default_dict=default_dict):
    default_nse = default_dict[eval_year]
    eval_nse = metric_dict[eval_year]
    reward = eval_nse - default_nse
    return reward

reward = evaluate_model(end_spinup_year, metric_dict=results_dict)

# Initialize Q-table
# Check whether the file exists so that old data isn't overwritten
if not os.path.isfile(q_table_output):
    q_table = pd.DataFrame(columns=list(velma_parameters.keys())+['Reward', f'{metric}', 'Year'])
    q_table.to_csv(q_table_output, mode='w', index=False)
else:
    q_table = pd.read_csv(q_table_output)
q_table.loc[len(q_table)] = parameter_values + [reward, results_dict[end_spinup_year], end_spinup_year]
q_table.loc[[len(q_table)-1]].to_csv(q_table_output, mode='a', header=False, index=False)
print('Q-table initialized.')

# Check if running_averages.csv exists; if not, need to initialize it
# Initialize table of parameters to record unique values and average NSE
if os.path.isfile(running_average_output):
    running_average = pd.read_csv(running_average_output)
else:
    running_average = pd.DataFrame(columns=list(velma_parameters.keys())+['Average_Reward']+['Data_Points'])
    running_average.loc[0] = parameter_values + [reward] + [1]
    running_average.to_csv(running_average_output, index=False)

print('Average reward table initialized.')

# Scale the data for use in GPR and Bayesian optimization
# Check whether the current parameter bounds or the min/max in the data should be used for scaling
scaler = MinMaxScaler()
param_bounds = ([[velma_parameters[param]['min'], velma_parameters[param]['max']] for param in velma_parameters])
X = running_average.iloc[:, :-2].values
X_min = X.min(axis=0).tolist()
X_max = X.max(axis=0).tolist()
param_space = []
for i, min in enumerate(X_min):
    if min < param_bounds[i][0]:
        data_min = min
    else:
        data_min = param_bounds[i][0]
    if X_max[i] > param_bounds[i][1]:
        data_max = X_max[i]
    else:
        data_max = param_bounds[i][1]
    param_space.append([data_min, data_max])
param_space = np.array(param_space).T
param_space = scaler.fit_transform(param_space)
param_space = param_space.T
param_space = [(bound[0], bound[1]) for bound in param_space]
X_scaled = scaler.transform(X)

# Gaussian Process Regression (surrogate model)
Y = running_average['Average_Reward'].values
kernel = DotProduct(sigma_0_bounds=(1e-20, 1e5)) + Matern(length_scale_bounds=(1e-12, 100)) + WhiteKernel(noise_level=1)
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=200)
gp_model = gp_model.fit(X_scaled, Y)

def objective(parameters):
    predicted_reward = gp_model.predict([parameters])[0]
    return -predicted_reward

# Run the below code in a loop for the rest of the years
for year in range(start_learning_year, end_year+1):
    print(f"Running VELMA for year: {year}")

    # Locations of folders for start data and end data
    start_data = f'{results_folder_root}/Results_{str(year - 1)}'
    end_data = f'{results_folder_root}/Results_{str(year)}'
    
    # If there are less than (3 * #parameters) unique data points, force random exploration of the parameter space
    # There's a probability (epsilon) of further exploration of the parameter space
    if len(running_average) < 3*len(velma_parameters) or random.random() < epsilon:
        if len(running_average) < 3*len(velma_parameters):
            print("Q-table is too sparse.")
        print(f"Forcing random exploration of parameter space.")    
        parameter_values = [round(np.random.uniform(velma_parameters[param]['min'], velma_parameters[param]['max']), 5) 
                            for param in velma_parameters]
        for idx, param in enumerate(velma_parameters.keys()):
            velma_parameters[param]['value'] = parameter_values[idx]
        # Formats parameters so they can modify the VELMA run
        parameter_modifiers = [f'--kv="{param}",{extended_velma_parameters[param]["value"]}' for param in extended_velma_parameters]
        
    # Print the parameter values, whether they were changed or not
    print(f"Values for {year} are:")
    for param in velma_parameters.keys():
        print(f"{velma_parameters[param]['name']}: {velma_parameters[param]['value']}")

    # Run VELMA for the current year
    run_velma(velma_parallel, allocated_memory, jar_path, xml_path, year-1, year, end_data, parameter_modifiers, start_data, max_processes=max_processes)
    if velma_parallel:
        results_path = f'{results_folder_root}/MULTI_{xml_name}/Results_{outlet_id}/DailyResults.csv'
    else:
        results_path = f'{results_folder_root}/{xml_name}/DailyResults.csv'

    # Calculate metric by year for the simulated results
    results_df = results_interpreter(results_path)
    grouped = align_group_data(observed_df, results_df)
    if metric == 'nse':
        results_dict = calculate_nse(grouped)
    elif metric == 'r2':
        results_dict = calculate_r2(grouped)
    metric_value = results_dict[year]
    print(f'{year} had {metric} = {round(metric_value, 2)}.')
    
    # Calculate reward
    reward = evaluate_model(year, metric_dict=results_dict)
    print(f"Reward for {year} was {round(reward, 3)}.")

    # Update the Q-table
    q_table.loc[len(q_table)] = parameter_values + [reward, metric_value, year]
    q_table.loc[[len(q_table)-1]].to_csv(q_table_output, mode='a', header=False, index=False)
    
    # Update the running average table
    mask = (q_table.iloc[:, :len(parameter_values)] == parameter_values).all(axis=1)
    matching_rows = q_table[mask]
    if not matching_rows.empty:
        # Calculate the average of the 'Reward' column in matching rows
        mean_reward = matching_rows['Reward'].mean()
        number_points = len(matching_rows)
        
        # Check if the parameter_values row already exists in running_average
        existing_row_mask = (running_average.iloc[:, :len(parameter_values)].eq(parameter_values).all(axis=1))

        if existing_row_mask.any():
            # Update the existing row with the new Average_NSE
            row_index = existing_row_mask.idxmax()  # Get the index of the first matching row
            running_average.at[row_index, 'Average_Reward'] = mean_reward
            running_average.at[row_index, 'Data_Points'] = number_points
        else:
            # Create a new row for the running average
            new_row = parameter_values + [mean_reward] + [1]
            running_average.loc[len(running_average)] = new_row

    # Save the running_averages to a .csv
    running_average.to_csv(running_average_output, index=False)

    # Update GPR model with new data
    X = running_average.iloc[:, :-2].values
    X_scaled = scaler.transform(X)
    Y = running_average['Average_Reward'].values
    gp_model.fit(X_scaled, Y)
    
    # Y_pred for R^2 score
    Y_pred = gp_model.predict(X_scaled)
    r2 = r2_score(Y, Y_pred)
    print(f'R^2 score for the GPR was {r2:.4f}')

    # Add in data for plotting
    X_plot = []
    for i, col in enumerate(X_scaled.T):
        X_added = []
        for j in range(len(col)-1):
            X_added.append(col[j])
            X_added.append((col[j] + col[j+1])/2)
        X_added.append(col[-1])
        X_plot.append(X_added)
    X_plot = np.array(X_plot)
    X_plot = X_plot.T
    Y_plot, sigma = gp_model.predict(X_plot, return_std=True)
    X_plot = scaler.inverse_transform(X_plot)

    # Plot the GPR
    for i, col in enumerate(X_plot.T):
        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(col)  # Sort by the values in the current column of X
        X_sorted = col[sorted_indices]  # Sort the current column
        Y_sorted = Y_plot[sorted_indices]
        sigma = sigma[sorted_indices]
        plt.plot(X.T[i], Y, 'r.', markersize=10, label='Observed Data')
        plt.plot(X_sorted, Y_sorted, 'b-', label='GPR Predictions')
        plt.fill_between(X_sorted, Y_sorted - 1.96 * sigma, Y_sorted + 1.96 * sigma,
                        alpha=0.2, color='blue', label='95% Confidence Interval')
        plt.title(f'Gaussian Process Regression Update for {parameter_names[i]}')
        plt.xlabel(f'{parameter_names[i]}')
        plt.ylabel('Rewards')
        plt.legend()
        plt.grid()
        plt.savefig(f'{figure_path}/gpr_{parameter_names[i]}')
        plt.close()
        
    # Permutation importance calculation
    perm_importance = permutation_importance(gp_model, X_scaled, Y, n_repeats=30, random_state=42)

    # Plot permutation importance
    plt.figure(figsize=(10, 6))
    plt.bar(parameter_names, perm_importance.importances_mean, yerr=perm_importance.importances_std)
    plt.title("Permutation Importance")
    plt.ylabel("Drop in R² Score")
    plt.savefig(f'{figure_path}/permutation_importance')
    plt.close()
    
    # Run Bayesian optimization to estimate the new parameter set
    result = gp_minimize(
        func=objective,
        dimensions=param_space,
        x0=[list(row) for row in X_scaled],  # Past parameter sets
        y0=-Y,  # Rewards (use numpy to turn negative because using minimization)
        n_calls=30,
        acq_func='EI',
    )
    
    best_scaled_params = result.x
    best_parameters = scaler.inverse_transform([best_scaled_params])[0]
    best_parameters = [round(param, 5) for param in best_parameters]

    # Update VELMA parameters for the next run
    for idx, param in enumerate(velma_parameters.keys()):
        velma_parameters[param]['value'] = best_parameters[idx]
    parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]
    parameter_modifiers = [f'--kv="{param},{extended_velma_parameters[param]["value"]}"' for param in extended_velma_parameters]  

    # Delete unnecessary data to save space
    if year > start_learning_year:
        shutil.rmtree(f'{results_folder_root}/Results_{str(year-1)}')
    if velma_parallel:
        shutil.rmtree(f'{results_folder_root}/MULTI_{xml_name}')
    else:
        shutil.rmtree(f'{results_folder_root}/{xml_name}')
        
print("Calibration process completed!")
print("Best predicted parameters:")
for param in velma_parameters.keys():
    print(f"{velma_parameters[param]['name']}: {velma_parameters[param]['value']}")    
