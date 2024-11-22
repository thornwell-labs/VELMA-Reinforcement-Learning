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


# Define the required run parameters and directories
start_year = 1987
start_learning_year = 1991
end_year = 2012
allocated_memory = "-Xmx10G"
jar_path = "C:/Users/thorn/OneDrive/Desktop/JVelma_dev-test_v003.jar"
working_directory = 'C:/Users/thorn/Documents/VELMA_Watersheds/Huge'
xml_name = 'WA_Huge30m_18Nov2024'
xml_path = f'{working_directory}/XML/{xml_name}.xml'
results_folder_root = f'{working_directory}/Results'
q_table_output = f'{results_folder_root}/q-table.csv'
running_average_output = f'{results_folder_root}/running-averages.csv'
figure_path = f'{results_folder_root}/Figures'
velma_parallel = True
epsilon = 0.2 # set the rate of random exploration here
# default_results file contains starting NSE values for comparison
default_results = f'{results_folder_root}/MULTI_WA_Huge30m_18Nov2024_default/Results_75524/AnnualHydrologyResults.csv'
# Required run parameters if velma_parallel is True
outlet_id = '75524'
max_processes = "6"


# Define starting and allowable VELMA parameter ranges
velma_parameters = {
    # '/calibration/VelmaCalibration.properties/setGroundwaterStorageFraction': {'name': 'GroundwaterStorageFraction','value': 0, 'min': 0.0, 'max': 0.15},
    '/calibration/VelmaCalibration.properties/f_ksv': {'name': 'VerticalKs', 'value': 0.0013, 'min': 0.001, 'max': 0.002},
    '/calibration/VelmaCalibration.properties/f_ksl': {'name': 'LateralKs', 'value': 0.00155, 'min': 0.001, 'max': 0.002},
    # '/soil/Lowland_Loam/soilColumnDepth': {'name': 'MediumSoilDepth','value': 1080, 'min': 1080, 'max': 1300},
    # '/soil/LowlandLoam_Shallow/soilColumnDepth': {'name': 'ShallowSoilDepth','value': 470, 'min': 470, 'max': 1080},
    # '/soil/LowlandLoam_Deep/soilColumnDepth': {'name': 'DeepSoilDepth','value': 1530, 'min': 1300, 'max': 3000},
    # '/soil/Lowland_Loam/surfaceKs': {'name': 'MediumKs','value': 891, 'min': 400, 'max': 2000},
    # '/soil/LowlandLoam_Shallow/surfaceKs': {'name': 'ShallowKs','value': 1200, 'min': 400, 'max': 2000},
    # '/soil/LowlandLoam_Deep/surfaceKs': {'name': 'DeepKs','value': 1200, 'min': 400, 'max': 2000}
    # Can add/remove parameters, but doing so necessitates manual changes to q-table and running-average table
}

parameter_names = [velma_parameters[param]["name"] for param in velma_parameters]

# List of parameter values for easy dataframe writing
parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]

# Formats parameters so they can modify the VELMA run
parameter_modifiers = [f'--kv="{param}",{velma_parameters[param]["value"]}' for param in velma_parameters]

# Run VELMA for spin-up years and output data at end to a folder
end_spinup_year = start_learning_year - 1
if os.path.exists(f'{results_folder_root}/Results_{end_spinup_year}'):
    print(f"Accessing data from previous spin-up years.")
else:
    print(f"Running spin-up years {start_year} to {end_spinup_year}.")
    if velma_parallel:
        command = ["java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaParallelCmdLine", xml_path,
                f"--maxProcesses={max_processes}",
                f'--kv="/calibration/VelmaInputs.properties/eyear",{end_spinup_year}',
                f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName",'
                f'{results_folder_root}/Results_{end_spinup_year}',
                *parameter_modifiers]
        command_str = ' '.join(command)
        subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('Spin-up years complete.')
    else:
        command = ["java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaSimulatorCmdLine", xml_path,
                f'--kv="/calibration/VelmaInputs.properties/eyear",{end_spinup_year}',
                f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName",'
                f'{results_folder_root}/Results_{end_spinup_year}',
                *parameter_modifiers]
        command_str = ' '.join(command)
        subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Initiate NSE using AnnualHydrologyResults
if velma_parallel:
    if not os.path.exists(f'{results_folder_root}/MULTI_{xml_name}_spinup'):
        os.rename(f'{results_folder_root}/MULTI_{xml_name}', f'{results_folder_root}/MULTI_{xml_name}_spinup')
else:
    if not os.path.exists(f'{results_folder_root}/{xml_name}_spinup'):
        os.rename(f'{results_folder_root}/{xml_name}', f'{results_folder_root}/{xml_name}_spinup')
if velma_parallel:
    simulated_results = pd.read_csv(f'{results_folder_root}/MULTI_{xml_name}_spinup/'
                                    f'/Results_{outlet_id}/AnnualHydrologyResults.csv')
else:
    simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}_spinup'
                                    f'/AnnualHydrologyResults.csv')
nse = simulated_results.loc[simulated_results['YEAR'] == end_spinup_year, 'Runoff_Nash-Sutcliffe_Coefficient']
nse = nse.values[0]
precip = simulated_results.loc[simulated_results['YEAR'] == end_spinup_year, 'Total_Rain+Melt(mm)']
precip = precip.values[0]

print(f'End spin-up year had NSE = {nse}.')

# Calculate the reward by comparing the new performance to the default performance
default_df = pd.read_csv(default_results, usecols=['YEAR', 'Runoff_Nash-Sutcliffe_Coefficient'])

def evaluate_model(eval_year, eval_nse, defaults=default_df):
    default_nse = default_df.loc[default_df['YEAR'] == eval_year, 'Runoff_Nash-Sutcliffe_Coefficient'].values[0]
    reward = eval_nse - default_nse
    return reward

reward = evaluate_model(end_spinup_year, nse)

# Initialize Q-table
# Check whether the file exists so that old data isn't overwritten
if not os.path.isfile(q_table_output):
    q_table = pd.DataFrame(columns=list(velma_parameters.keys())+['Reward', 'NSE', 'Year', 'Precip'])
    q_table.to_csv(q_table_output, mode='w', index=False)
else:
    q_table = pd.read_csv(q_table_output)
q_table.loc[len(q_table)] = parameter_values + [reward, nse, end_spinup_year, precip]
q_table.loc[[len(q_table)-1]].to_csv(q_table_output, mode='a', header=False, index=False)
print('Q-table initialized.')

# Check if running_averages.csv exists; if not, need to initialize it
# Initialize table of parameters to record unique values and average NSE
if os.path.isfile(running_average_output):
    running_average = pd.read_csv(running_average_output)
else:
    running_average = pd.DataFrame(columns=list(velma_parameters.keys())+['Average_Reward'])
    running_average.loc[0] = parameter_values + [reward]
    running_average.to_csv(running_average_output, index=False)

print('Average reward table initialized.')

# Scale the data for use in GPR and Bayesian optimization
# Check whether the current parameter bounds or the min/max in the data should be used for scaling
scaler = MinMaxScaler()
param_bounds = ([[velma_parameters[param]['min'], velma_parameters[param]['max']] for param in velma_parameters])
X = running_average.iloc[:, :-1].values
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
        parameter_modifiers = [f'--kv="{param}",{velma_parameters[param]["value"]}' for param in velma_parameters]
        
    # Print the parameter values, whether they were changed or not
    print(f"Values for {year} are:")
    for param in velma_parameters.keys():
        print(f"{velma_parameters[param]['name']}: {velma_parameters[param]['value']}")

    # Run VELMA for the current year
    if velma_parallel:
        command = [
             "java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaParallelCmdLine", xml_path,
             f"--maxProcesses={max_processes}", f'--kv="/calibration/VelmaInputs.properties/syear",{str(year)}',
             f'--kv="/calibration/VelmaInputs.properties/eyear",{str(year+1)}',
             f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName,{end_data}"',
             f'--kv="/startups/VelmaStartups.properties/setStartStateSpatialDataLocationName,{start_data}"',
             *parameter_modifiers]
        command_str = ' '.join(command)
        subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        command = [
             "java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaSimulatorCmdLine", xml_path,
             f'--kv="/calibration/VelmaInputs.properties/syear",{year}',
             f'--kv="/calibration/VelmaInputs.properties/eyear",{year+1}',
             f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName,{end_data}"',
             f'--kv="/startups/VelmaStartups.properties/setStartStateSpatialDataLocationName,{start_data}"',
             *parameter_modifiers]
        command_str = ' '.join(command)
        subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Find NSE using AnnualHydrologyResults
    if velma_parallel:
        simulated_results = pd.read_csv(f'{results_folder_root}/MULTI_{xml_name}/'
                                        f'Results_{outlet_id}/AnnualHydrologyResults.csv')
    else:
        simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}/AnnualHydrologyResults.csv')
    nse = simulated_results.loc[simulated_results['YEAR'] == year, 'Runoff_Nash-Sutcliffe_Coefficient']
    nse = nse.values[0]
    precip = simulated_results.loc[simulated_results['YEAR'] == year, 'Total_Rain+Melt(mm)']
    precip = precip.values[0]
    
    print(f"{year} had NSE of {round(nse, 2)}.")
    
    # Calculate reward
    reward = evaluate_model(year, nse)
    print(f"Reward for {year} was {round(reward, 3)}.")

    # Update the Q-table
    q_table.loc[len(q_table)] = parameter_values + [reward, nse, year, precip]
    q_table.loc[[len(q_table)-1]].to_csv(q_table_output, mode='a', header=False, index=False)
    
    # Update the running average table
    mask = (q_table.iloc[:, :len(parameter_values)] == parameter_values).all(axis=1)
    matching_rows = q_table[mask]
    if not matching_rows.empty:
        # Calculate the average of the 'Reward' column in matching rows
        mean_reward = matching_rows['Reward'].mean()
        
        # Check if the parameter_values row already exists in running_average
        existing_row_mask = (running_average.iloc[:, :len(parameter_values)].eq(parameter_values).all(axis=1))

        if existing_row_mask.any():
            # Update the existing row with the new Average_NSE
            row_index = existing_row_mask.idxmax()  # Get the index of the first matching row
            running_average.at[row_index, 'Average_Reward'] = mean_reward
        else:
            # Create a new row for the running average
            new_row = parameter_values + [mean_reward]
            running_average.loc[len(running_average)] = new_row

    # Save the running_averages to a .csv
    running_average.to_csv(running_average_output, index=False)

    # Update GPR model with new data
    X = running_average.iloc[:, :-1].values
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
    parameter_modifiers = [f'--kv="{param},{velma_parameters[param]["value"]}"' for param in velma_parameters]  

    # Delete unnecessary data to save space
    if year > start_learning_year+2:
        shutil.rmtree(f'{results_folder_root}/Results_{str(year-2)}')
    if velma_parallel:
        shutil.rmtree(f'{results_folder_root}/MULTI_{xml_name}')
    else:
        shutil.rmtree(f'{results_folder_root}/{xml_name}')
        
print("Calibration process completed!")
print("Best predicted parameters:")
for param in velma_parameters.keys():
    print(f"{velma_parameters[param]['name']}: {velma_parameters[param]['value']}")    
