"""
Main file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, DotProduct, WhiteKernel, RationalQuadratic
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import random
from sklearn.inspection import permutation_importance
from rls_functions import results_interpreter, align_group_data, calculate_apes, calculate_nse, \
calculate_r2, calculate_soar, calculate_soar_summer, evaluate_model, run_velma, row_col_to_index, \
log_message, log_with_timestamp, start_timer, elapsed_time, generate_lhs_param_sets
from resample import resample_xml
from divide_catchments import divide_catchments
import xml.etree.ElementTree as ET
import pickle

# Carefully define the following required run parameters and directories
start_year = 1987  # Start the model spin-up from this year
start_learning_year = 1991  # Start model calibration from this year
end_year = 2021
allocated_memory = "-Xmx2G"
jar_path = "C:/Users/thorn/OneDrive/Desktop/JVelma_dev-test_v009.jar"
working_directory = 'C:/Users/thorn/Documents/VELMA_Watersheds/Big_Beef'
xml_name = 'WA_BigBeef30m_5Dec2025'  # Do not include .xml extension in this name
xml_path = f'{working_directory}/XML/{xml_name}.xml'  # Set up your working directory so this line works as-is
calibration_results_path = f'{working_directory}/Results/RL_Testing/RL_Testing_3_year_nse_cap'  # All the calibration results should write to a sub-folder. Set the xml to write to this folder, too.
log_file_name = 'RL_Testing_3_year_nse_cap.txt'  # Name this file something descriptive
default_path = f'{working_directory}/Results/MULTI_WA_BigBeef30m_5Dec2025/Results_23023/DailyResults.csv'  # Compare to these results. Must point to a DailyResults.csv file
calibration_data = 'Runoff_All(mm/day)_Delineated_Average'  # must exactly match a column name in the DailyResults.csv file
start_obs_data_year = 1987  # Enter the year the observed data starts from (be sure to check the observed data file)
observed_file = f'{working_directory}/DataInputs30m/m_7_Observed/USGS12069550_BigBeef_1987-2021_streamflow.csv'

# Specify number of model years that constitute a data point and desired number of total data points
n_years_per_point = 3
n_data_points = 100

# Specify number of data points in initial exploration period and rate of exploration
# Recommend the initial exploration period should be at least (3 * number of parameters)
n_initial_exploration = 35
epsilon = 0.2

# Specify the following weights for Aggregate Performance Efficiency Statistic (APES)
soar_weight = 0.5
nse_weight = 1.0
summer_soar_weight = 0.7
r2_weight = 0.2

# Specify downscaling and VELMA parallel parameters
velma_parallel = False
max_processes = "4"
outlet_id = '23023'
downscaling = True
downscaling_factor = 20

if velma_parallel == True and downscaling == True:
    divide_catchments_flag = True
    number_catchments = 4
    crs = 'EPSG:26910'

# Define starting and allowable VELMA parameter ranges in this dictionary
# 'value' must match value used in default results
velma_parameters = {
    '/calibration/VelmaCalibration.properties/setGroundwaterStorageFraction': {'name': 'GroundwaterStorageFraction','value': 0.0, 'min': 0.0, 'max': 0.20},
    # 'soil/Medium_CN24/ksVerticalExponentialDecayFactor': {'name': 'Medium VerticalKs', 'value': 0.0013, 'min': 0.001, 'max': 0.002},
    # '/soil/Shallow_CN24/ksVerticalExponentialDecayFactor': {'name': 'Shallow VerticalKs', 'value': 0.0013, 'min': 0.001, 'max': 0.002},
    # '/soil/Deep_CN24/ksVerticalExponentialDecayFactor': {'name': 'Deep VerticalKs', 'value': 0.0013, 'min': 0.001, 'max': 0.002},
    # 'soil/Medium_CN24/ksLateralExponentialDecayFactor': {'name': 'Medium LateralKs', 'value': 0.00155, 'min': 0.001, 'max': 0.002},
    # '/soil/Shallow_CN24/ksLateralExponentialDecayFactor': {'name': 'Shallow LateralKs', 'value': 0.00155, 'min': 0.001, 'max': 0.002},
    # '/soil/Deep_CN24/ksLateralExponentialDecayFactor': {'name': 'Deep LateralKs', 'value': 0.00155, 'min': 0.001, 'max': 0.002},
    '/soil/Medium_CN24/soilColumnDepth': {'name': 'MediumSoilDepth','value': 1080, 'min': 1080, 'max': 1300},
    '/soil/Shallow_CN24/soilColumnDepth': {'name': 'ShallowSoilDepth','value': 470, 'min': 470, 'max': 1080},
    '/soil/Deep_CN24/soilColumnDepth': {'name': 'DeepSoilDepth','value': 1300, 'min': 1300, 'max': 3000},
    '/soil/Medium_CN24/surfaceKs': {'name': 'MediumKs','value': 1200, 'min': 200, 'max': 1800},
    '/soil/Shallow_CN24/surfaceKs': {'name': 'ShallowKs','value': 1200, 'min': 200, 'max': 1800},
    '/soil/Deep_CN24/surfaceKs': {'name': 'DeepKs','value': 1200, 'min': 200, 'max': 1800}
    # Can add/remove parameters
}

# ------------------------------------------ Do not edit anything below this line ---------------------------------------------

# Basic checks on the input directories
for directory in [jar_path, working_directory, xml_path, default_path, observed_file]:
    if not os.path.exists(directory):
        print(f'FATAL WARNING: {directory} does not exist. Killing program. \nCheck directories and restart script.')
        exit()
if not os.path.exists(calibration_results_path):
    os.mkdir(calibration_results_path)
figure_path = f'{calibration_results_path}/Figures'
if not os.path.exists(figure_path):
    os.mkdir(figure_path)
print('Input directories are valid.')

# Outdated file / folder cleaning
for year in range(start_learning_year, end_year+1):
    if os.path.exists(f'{calibration_results_path}/Results_{year}'):
        print(f'Removing outdated results folder Results_{year}')
        shutil.rmtree(f'{calibration_results_path}/Results_{year}')

# Initialize log file
logfile = os.path.join(calibration_results_path, log_file_name)
log_with_timestamp("Starting calibration script", logfile)
log_message(f"Years per data point: {n_years_per_point}", logfile)
log_message(f"Length of initial exploration: {n_initial_exploration} data points", logfile)
log_message(f"Chance of random exploration (epsilon): {epsilon}", logfile)
log_message("Parameters used: ", logfile)
for key, info in velma_parameters.items():
    log_message(f"[PARAM] {info['name']} (key={key}) | value={info['value']} | range=({info['min']}, {info['max']})", logfile)


# Handle downscaling in this block, if downscaling is on
if downscaling == True:
    if not os.path.exists(f'{working_directory}/XML/{xml_name}_resampled_{downscaling_factor}.xml'):
        print('Performing downscaling.')
        start_downscaling = start_timer()
        new_xml = resample_xml(xml_path, 'resampled', downscale_factor=downscaling_factor, plot_dem=False, overwrite=True, plot_hist=False, weights=None, change_disturbance_fraction=True)
        downscaling_elapsed = elapsed_time(start_downscaling)
        log_with_timestamp(f'Downscaling took {downscaling_elapsed} seconds. Downscaling factor is {downscaling_factor}.', logfile)
        tree = ET.parse(new_xml)
        new_downscaled_file = True
    else:
        print('Accessing previous downscaled model version.')
        log_message(f'Previous downscaled model version was used. Downscaling factor is {downscaling_factor}.', logfile)
        new_downscaled_file = False
        tree = ET.parse(f'{working_directory}/XML/{xml_name}_resampled_{downscaling_factor}.xml')
    
    xml_path = f'{working_directory}/XML/{xml_name}_resampled_{downscaling_factor}.xml'
    xml_name = xml_name+f'_resampled_{downscaling_factor}'
    
    # Find the x, y location of the new outlet
    root = tree.getroot()
    input_props = root.find(".//VelmaInputs.properties")
    startup_props = root.find(".//VelmaStartups.properties")
    input_root_name = startup_props.find("inputDataLocationRootName").text.strip()
    input_dir_name  = startup_props.find("inputDataLocationDirName").text.strip()
    dem_path = input_props.find("input_dem").text.strip()
    # strip leading './' so the join works cleanly
    dem_path = dem_path.lstrip("./\\")
    asc_file_path = os.path.join(input_root_name, input_dir_name, dem_path)
    asc_file_path = os.path.normpath(asc_file_path)
    col = int(input_props.find("outx").text.strip())
    row = int(input_props.find("outy").text.strip())
    
    # If velma_parallel is also True, divide catchments and add the outlet list to the xml
    if velma_parallel == True and new_downscaled_file == True:
        print('Performing catchment division.')
        new_outlets = divide_catchments(asc_file_path, col, row, num_processors=int(max_processes), num_subbasins=number_catchments, method='equal', crs=crs, is_plot=False)
        # Replace the reach outlet list with the new outlet list
        input_props.find("initialReachOutlets").text = " ".join(map(str, new_outlets))
        input_props.find("run_index").text = xml_name
        tree.write(new_xml)
    
    # Update the outlet ID with the downscaled version
    new_outlet = row_col_to_index(asc_file_path, row, col)
    outlet_id = str(new_outlet)
else:
    log_message('No downscaling used.', logfile)

# Force the Results location to the specified results folder
# Also ensure the simulation run name matches the xml name
tree = ET.parse(xml_path)
root = tree.getroot()
elem = root.find(
    ".//VelmaInputs.properties/initializeOutputDataLocationRoot"
)
elem.text = calibration_results_path
run_elem = root.find(".//VelmaInputs.properties/run_index")
run_elem.text = xml_name
tree.write(xml_path)

# Read in the observed data and assign an index by date
observed_df = pd.read_csv(observed_file, usecols=[0], header=None, names=[calibration_data])
start_date = f'1/1/{start_obs_data_year}'
date_range = pd.date_range(start=start_date, periods=len(observed_df), freq='D')
observed_df['Date'] = date_range
observed_df.set_index('Date', inplace=True)

weights = [nse_weight, r2_weight, summer_soar_weight, soar_weight]

# Compare observed data with baseline results and create dataframes that contain APES metrics by year
default_df = results_interpreter(default_path, calibration_data)
default_data = align_group_data(observed_df, default_df)
default_nse_dict = calculate_nse(default_data, calibration_data)
default_r2_dict = calculate_r2(default_data, calibration_data)
default_soar_summer_dict = calculate_soar_summer(default_data, calibration_data)
default_soar_dict = calculate_soar(default_data, calibration_data)
default_results = [default_nse_dict, default_r2_dict, default_soar_summer_dict, default_soar_dict]

for metric, dict in zip(['nse', 'r2', 'soar_summer', 'soar'], default_results):
    df = pd.DataFrame.from_dict(dict, orient='index', columns=[metric])
    df.index.name = 'Year'
    output_file = f'default_{metric}.csv'
    df.to_csv(f'{calibration_results_path}/{output_file}')
    print(f'Successfully wrote default {metric} to file {calibration_results_path}/{output_file}.')

parameter_names = [velma_parameters[param]["name"] for param in velma_parameters]
parameter_exact_names = list(velma_parameters.keys())

# List of parameter values for easy dataframe writing
parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]

# Format parameters so they can modify the VELMA run
extended_velma_parameters = velma_parameters.copy()

# This section is only for modification of VELMA run parameters using extended soil types with C/N ratios
soil_type_dict = {
    'CN24': ['CN12', 'CN17'],
}
for soil_type, ratios in soil_type_dict.items():
    for parameter in parameter_exact_names:
        if soil_type in parameter:
            for ratio in ratios:
                extended_velma_parameters[parameter.replace(soil_type, ratio)] = extended_velma_parameters[parameter]

parameter_modifiers = [f'--kv="{param}",{extended_velma_parameters[param]["value"]}' for param in extended_velma_parameters]

# Run VELMA for spin-up years and output data at end to a folder
end_spinup_year = start_learning_year - 1
end_data = f'{calibration_results_path}/Results_{str(end_spinup_year)}'
if os.path.exists(end_data):
    print(f"Accessing data from previous spin-up years.")
else:
    print(f"Running spin-up years {start_year} to {end_spinup_year}.")
    log_with_timestamp('Starting spin-up model run', logfile)
    start_spinup = start_timer()
    run_velma(velma_parallel, allocated_memory, jar_path, xml_path, start_year, end_spinup_year, end_data, parameter_modifiers, start_data=None, max_processes=max_processes)
    spinup_elapsed = elapsed_time(start_spinup)
    log_with_timestamp(f'Spin-up model run complete. Took {spinup_elapsed} seconds', logfile)
    print('Spin-up years complete.')

# Check for spinup folder and rename if necessary
if velma_parallel:
    if not os.path.exists(f'{calibration_results_path}/MULTI_{xml_name}_spinup'):
        os.rename(f'{calibration_results_path}/MULTI_{xml_name}', f'{calibration_results_path}/MULTI_{xml_name}_spinup')
else:
    if not os.path.exists(f'{calibration_results_path}/{xml_name}_spinup'):
        os.rename(f'{calibration_results_path}/{xml_name}', f'{calibration_results_path}/{xml_name}_spinup')
if velma_parallel:
    daily_results_path = f'{calibration_results_path}/MULTI_{xml_name}_spinup/Results_{outlet_id}/DailyResults.csv'
else:
    daily_results_path = f'{calibration_results_path}/{xml_name}_spinup/DailyResults.csv'

# Calculate metrics by year for the simulated results - end up with a list of dictionaries
results_df = results_interpreter(daily_results_path, calibration_data)
simulated_data = align_group_data(observed_df, results_df)
results_nse_dict = calculate_nse(simulated_data, calibration_data)
results_r2_dict = calculate_r2(simulated_data, calibration_data)
results_soar_summer_dict = calculate_soar_summer(simulated_data, calibration_data)
results_soar_dict = calculate_soar(simulated_data, calibration_data)
results = [results_nse_dict, results_r2_dict, results_soar_summer_dict, results_soar_dict]

# Calculate APES score
apes_score = calculate_apes(years=[end_spinup_year], results=results, weights=weights)

print(f'End spin-up year had NSE = {results_nse_dict[end_spinup_year]}, '
      f'R^2 = {results_r2_dict[end_spinup_year]}, '
      f'Summertime corrected SOAR = {results_soar_summer_dict[end_spinup_year]}, '
      f'and overall corrected SOAR = {results_soar_dict[end_spinup_year]}, '
      f'for an APES score of {apes_score}.')

# Initiate scoring record
scoring_data = [dict[end_spinup_year] for dict in results]
scoring_data.append(apes_score)
scoring_data.append(end_spinup_year)
scoring_df = pd.DataFrame(data=[scoring_data], columns=['NSE', 'R2', 'Summer SOAR', 'SOAR', 'APES Score', 'Year'])
scoring_df = scoring_df.set_index('Year')
scoring_df.to_csv(f'{calibration_results_path}/apes_scores.csv')

reward = evaluate_model(end_spinup_year, results=results, default_results=default_results, weights=weights)

# Initialize Q-table
# Check whether the file exists so that old data isn't overwritten
q_table_output = f'{calibration_results_path}/q-table.csv'
if not os.path.isfile(q_table_output):
    q_table = pd.DataFrame(columns=parameter_exact_names+['Reward', 'APES_Score'])
    q_table.to_csv(q_table_output, mode='w', index=False)
else:
    q_table = pd.read_csv(q_table_output)
q_table.loc[len(q_table)] = parameter_values + [reward, apes_score]
q_table.loc[[len(q_table)-1]].to_csv(q_table_output, mode='a', header=False, index=False)
print('Q-table initialized.')

# Check if running_averages.csv exists; if not, need to initialize it
# Initialize table of parameters to record unique values and average NSE
running_average_output = f'{calibration_results_path}/running-averages.csv'
if os.path.isfile(running_average_output):
    running_average = pd.read_csv(running_average_output)
else:
    running_average = pd.DataFrame(columns=parameter_exact_names+['Average_Reward']+['Data_Points'])
    running_average.loc[0] = parameter_values + [reward] + [1]
    running_average.to_csv(running_average_output, index=False)

print('Average reward table initialized.')

# Scale the data for use in GPR
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

# Generate samples using LHS for the initial exploration phase
param_names, lhs_sets = generate_lhs_param_sets(velma_parameters, n_initial_exploration)
if len(q_table) < n_initial_exploration:
    sample = lhs_sets[len(q_table)]
    for i, p in enumerate(parameter_exact_names):
        velma_parameters[p]['value'] = float(sample[i])
        parameter_values[i] = float(sample[i])
    parameter_modifiers = [f'--kv="{param}",{extended_velma_parameters[param]["value"]}' for param in extended_velma_parameters]

# Gaussian Process Regression (surrogate model)
Y = running_average['Average_Reward'].values
kernel = Matern(nu=1.5) + RationalQuadratic()
# kernel = DotProduct(sigma_0_bounds=(1e-10, 1e5)) + Matern(length_scale_bounds=(1e-12, 100)) + WhiteKernel(noise_level=1)

gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, alpha=0.05)
log_with_timestamp(f"GPR kernel: {gp_model.kernel}", logfile)

# If already past initial exploration, fit the GPR and randomize the parameters
if len(q_table) > n_initial_exploration:
    gp_model.fit(X_scaled, Y)
    
    # Y_pred for R^2 score
    Y_pred = gp_model.predict(X_scaled)
    r2 = r2_score(Y, Y_pred)
    print(f'R^2 score for the GPR was {r2:.4f}')
    log_message(f'R^2 score for the GPR was {r2:.4f}', logfile)
    parameter_values = [round(np.random.uniform(velma_parameters[param]['min'], velma_parameters[param]['max']), 5) 
                            for param in velma_parameters]
    for idx, param in enumerate(parameter_exact_names):
        velma_parameters[param]['value'] = parameter_values[idx]
    # Formats parameters so they can modify the VELMA run
    parameter_modifiers = [f'--kv="{param}",{extended_velma_parameters[param]["value"]}' for param in extended_velma_parameters]
    
gpr_path = os.path.join(calibration_results_path, 'gpr_model.pkl')
with open(gpr_path, "wb") as f:
    pickle.dump(gp_model, f)

def objective(parameters):
    predicted_reward = gp_model.predict([parameters])[0]
    return -predicted_reward

# Run the below code in a loop to collect all data points
while len(q_table) <= n_data_points:
    if year+n_years_per_point > end_year:
        log_message('Starting model from start learning year.', logfile)
        year = start_learning_year
        
    print(f"Collecting data point number {len(q_table)} out of {n_data_points}.")
    log_with_timestamp(f"Collecting data point number {len(q_table)} out of {n_data_points}.", logfile)
    start_velma = start_timer()
    
    # Delete working results
    if velma_parallel:
        if os.path.exists(f'{calibration_results_path}/MULTI_{xml_name}'):
            shutil.rmtree(f'{calibration_results_path}/MULTI_{xml_name}')
    else:
        if os.path.exists(f'{calibration_results_path}/{xml_name}'):
            shutil.rmtree(f'{calibration_results_path}/{xml_name}')    

    # Locations of folders for start data and end data
    start_data = f'{calibration_results_path}/Results_{str(year - 1)}'
    end_data = f'{calibration_results_path}/Results_{str(year+n_years_per_point-1)}'
    if os.path.exists(end_data):
        shutil.rmtree(end_data)    
    
    # If initial data collection is not complete, force exploration of the parameter space
    if len(q_table) < n_initial_exploration:
        print(f"Q-table is too sparse. Need {n_initial_exploration} points in initial exploration.")
        sample = lhs_sets[len(q_table)]
        for i, p in enumerate(parameter_exact_names):
            velma_parameters[p]['value'] = float(sample[i])
            parameter_values[i] = float(sample[i])
        parameter_modifiers = [f'--kv="{param}",{extended_velma_parameters[param]["value"]}' for param in extended_velma_parameters]
    
    # After initial exploration, there's a probability (epsilon) of further random exploration of the parameter space
    if len(q_table) >  n_initial_exploration and random.random() < epsilon:  
        print(f"Forcing random exploration of parameter space.")    
        parameter_values = [round(np.random.uniform(velma_parameters[param]['min'], velma_parameters[param]['max']), 5) 
                            for param in velma_parameters]
        for idx, param in enumerate(parameter_exact_names):
            velma_parameters[param]['value'] = parameter_values[idx]
        # Formats parameters so they can modify the VELMA run
        parameter_modifiers = [f'--kv="{param}",{extended_velma_parameters[param]["value"]}' for param in extended_velma_parameters]
        
    # Print the parameter values, whether they were changed or not
    print(f"Values for {year} to {year+n_years_per_point-1} are:")
    for param in parameter_exact_names:
        print(f"{velma_parameters[param]['name']}: {velma_parameters[param]['value']}")

    # Run VELMA for the current data point
    run_velma(velma_parallel, allocated_memory, jar_path, xml_path, year, year+n_years_per_point-1, end_data, parameter_modifiers, start_data, max_processes=max_processes)
    if velma_parallel:
        daily_results_path = f'{calibration_results_path}/MULTI_{xml_name}/Results_{outlet_id}/DailyResults.csv'
    else:
        daily_results_path = f'{calibration_results_path}/{xml_name}/DailyResults.csv'
    velma_elapsed = elapsed_time(start_velma)
    log_with_timestamp(f'VELMA run complete. Took {velma_elapsed} seconds.', logfile)
    
    # Delete unnecessary data to save space
    if year > start_learning_year:
        shutil.rmtree(start_data)

    # Calculate metric by year for the simulated results
    results_df = results_interpreter(daily_results_path, calibration_data)
    simulated_data = align_group_data(observed_df, results_df)
    
    # If there is no observed data in these year(s), need to re-run the model
    if len(simulated_data) == 0:
        print(f'The year(s) {year} to {year+n_years_per_point-1} had no observed data. Re-running model without updated parameters.')
        log_message('No observed data. Re-running model without updated parameters', logfile)
        year = year + n_years_per_point
        continue
    
    results_nse_dict = calculate_nse(simulated_data, calibration_data)
    results_r2_dict = calculate_r2(simulated_data, calibration_data)
    results_soar_summer_dict = calculate_soar_summer(simulated_data, calibration_data)
    results_soar_dict = calculate_soar(simulated_data, calibration_data)
    results = [results_nse_dict, results_r2_dict, results_soar_summer_dict, results_soar_dict]
    apes_score = calculate_apes(years=[year], results=results, weights=weights)
    
    # If APES score is missing for any other reason (for example, no summertime data), need to re-run the model
    if np.isnan(apes_score):
        print(f'The year(s) {year} to {year+n_years_per_point-1} did not have a valid APES score. Re-running model without updated parameters.')
        log_message('No valid APES score. Re-running model without updated parameters', logfile)
        year = year + n_years_per_point
        continue

    print(f'{year} had APES score = {round(apes_score, 2)}.')
    
    # Update the APES score spreadsheet
    scoring_data = [dict[year] for dict in results]
    scoring_data.append(apes_score)
    scoring_df.loc[year] = scoring_data
    scoring_df.to_csv(f'{calibration_results_path}/apes_scores.csv')
    
    # Calculate reward
    reward = evaluate_model(year, results=results, default_results=default_results, weights=weights)
    print(f"Reward for {year} to {year+n_years_per_point-1} was {round(reward, 3)}.")

    # Update the Q-table
    q_table.loc[len(q_table)] = parameter_values + [reward, apes_score]
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
            # Update the existing row with the new average reward
            row_index = existing_row_mask.idxmax()  # Get the index of the first matching row
            running_average.at[row_index, 'Average_Reward'] = mean_reward
            running_average.at[row_index, 'Data_Points'] = number_points
        else:
            # Create a new row for the running average
            new_row = parameter_values + [mean_reward] + [1]
            running_average.loc[len(running_average)] = new_row

    start_update = start_timer()
    # Save the running_averages to a .csv
    running_average.to_csv(running_average_output, index=False)
    if len(q_table) >= n_initial_exploration:
        # Update GPR model with new data
        X = running_average.iloc[:, :-2].values
        X_scaled = scaler.transform(X)
        Y = running_average['Average_Reward'].values
        gp_model.fit(X_scaled, Y)
        with open(gpr_path, "wb") as f:
            pickle.dump(gp_model, f)
        
        # Y_pred for R^2 score
        Y_pred = gp_model.predict(X_scaled)
        r2 = r2_score(Y, Y_pred)
        print(f'R^2 score for the GPR was {r2:.4f}')
        log_message(f'R^2 score for the GPR was {r2:.4f}', logfile)

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
            plt.plot(X.T[i], Y, 'r.', markersize=10, label='Observed Reward')
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
            
        # Permutation importance calculation and dump to csv
        perm_importance = permutation_importance(gp_model, X_scaled, Y, n_repeats=30, random_state=42)
        perm_df = pd.DataFrame({
        "feature": parameter_exact_names,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std,
        })
        perm_df.to_csv(os.path.join(calibration_results_path, "permutation_importance.csv"), index=False)

        # Plot permutation importance
        plt.figure(figsize=(10, 6))
        plt.bar(parameter_names, perm_importance.importances_mean, yerr=perm_importance.importances_std)
        plt.title("Permutation Importance")
        plt.ylabel("Drop in R² Score")
        plt.savefig(f'{figure_path}/permutation_importance')
        plt.close()

        # Run differential evolution to estimate the new parameter set
        result = differential_evolution(
            func=objective,
            bounds=param_space,
            maxiter=100,
            popsize=10,
            tol=1e-8,
            mutation=(0.8, 1.2),
            recombination=0.2,
            seed=42,
            polish=True,
            disp=False,
        )
        best_scaled_params = result.x
        best_parameters = scaler.inverse_transform([best_scaled_params])[0].tolist()

        best_parameters = [round(param, 5) for param in best_parameters]
        
        update_elapsed = elapsed_time(start_update)
        log_with_timestamp(f'GPR update, plotting, and differential evolution complete. Took {update_elapsed} seconds.', logfile)

        # Update VELMA parameters for the next run
        for idx, param in enumerate(parameter_exact_names):
            velma_parameters[param]['value'] = best_parameters[idx]
        parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]
        for p in parameter_exact_names:
            if p in extended_velma_parameters:
                extended_velma_parameters[p]['value'] = velma_parameters[p]['value']
        parameter_modifiers = [f'--kv="{param},{extended_velma_parameters[param]["value"]}"' for param in extended_velma_parameters] 
    else:
        log_message(f'Initial exploration phase: {len(q_table)} out of {n_initial_exploration} points. No GPR fitting performed.', logfile)

    year = year + n_years_per_point


print("Calibration process completed!")
print("Parameters with best performance:")

# Choose the parameters with the highest actual average reward
best_row = running_average.loc[running_average['Average_Reward'].idxmax(), parameter_exact_names]
for param in parameter_exact_names:
    velma_parameters[param]['value'] = best_row[param]

for param in parameter_exact_names:
    print(f"{velma_parameters[param]['name']}: {velma_parameters[param]['value']}")   
     
log_with_timestamp(f'Calibration process completed!', logfile)
log_message(
    "Parameters with best performance:\n" +
    "\n".join(
        f"{velma_parameters[p]['name']}: {velma_parameters[p]['value']}"
        for p in velma_parameters
    ),
    logfile
)
