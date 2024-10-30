import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import shutil
from skopt import gp_minimize
from skopt.space import Real
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import MinMaxScaler


# Define the required run parameters and directories
start_year = 1969
start_learning_year = 1972
end_year = 2005
allocated_memory = "-Xmx20G"
jar_path = "C:/Users/thorn/OneDrive/Desktop/JVelma_CW3-495_v02_GBox.jar"
working_directory = 'C:/Users/thorn/Documents/Automated_Calibration/WS10'
xml_name = 'OR_BR_ws10_14Oct24'
xml_path = f'{working_directory}/XMLs/{xml_name}.xml'
results_folder_root = f'{working_directory}/Results'
q_table_output = f'{results_folder_root}/q-table.csv'
running_average_output = f'{results_folder_root}/running-averages.csv'
figure_path = f'{results_folder_root}/Figures'
velma_parallel = False
# default_results file contains starting NSE values for comparison
default_results = f'{results_folder_root}/MULTI_WS10_DefaultRun/Results_2932/AnnualHydrologyResults.csv'

# Required run parameters if velma_parallel is True
outlet_id = '2932'
max_processes = "6"

# Define starting and allowable VELMA parameter ranges
velma_parameters = {
    '/calibration/VelmaInputs.properties/be': {'name': 'be','value': 1.0, 'min': 0.7, 'max': 1.2},
    '/soil/Soil1_sandy_loam/surfaceKs': {'name': 'Soil1_surfaceKs','value': 600, 'min': 500, 'max': 1200}
    # '/': {'value': 1.0, 'min': 0.5, 'max': 1.5},
    # Can add/remove parameters
}

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
if not os.path.exists(f'{results_folder_root}/{xml_name}_spinup'):
    os.rename(f'{results_folder_root}/{xml_name}', f'{results_folder_root}/{xml_name}_spinup')
if velma_parallel:
    simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}_spinup/'
                                    f'/Results_{outlet_id}/AnnualHydrologyResults.csv')
else:
    simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}_spinup'
                                    f'/AnnualHydrologyResults.csv')
nse = simulated_results.loc[simulated_results['YEAR'] == end_spinup_year, 'Runoff_Nash-Sutcliffe_Coefficient']
nse = nse.values[0]

print(f'End spin-up year had NSE = {nse}.')

# Calculate the reward by comparing the new performance to the default performance
default_df = pd.read_csv(default_results, usecols=['YEAR', 'Runoff_Nash-Sutcliffe_Coefficient'])

def evaluate_model(eval_year, eval_nse, defaults=default_df):
    default_nse = default_df.loc[default_df['YEAR'] == eval_year, 'Runoff_Nash-Sutcliffe_Coefficient'].values[0]
    reward = 100*(eval_nse - default_nse)
    return reward
    
reward = evaluate_model(end_spinup_year, nse)

# Initialize Q-table
# Check whether the file exists so that old data isn't overwritten
if not os.path.isfile(q_table_output):
    q_table = pd.DataFrame(columns=list(velma_parameters.keys())+['Reward', 'NSE', 'Year'])
    q_table.to_csv(q_table_output, mode='w', index=False)
else:
    q_table = pd.read_csv(q_table_output)
q_table.loc[len(q_table)] = parameter_values + [reward, nse, end_spinup_year]
q_table.loc[[len(q_table)-1]].to_csv(q_table_output, mode='a', header=False, index=False)
print('Q-table initialized.')

# Check if running_averages.csv exists; if not, need to initialize it
# Initialize table of parameters to record unique values and average NSE
if os.path.exists(running_average_output):
    running_average = pd.read_csv(running_average_output)
else:
    running_average = pd.DataFrame(columns=list(velma_parameters.keys())+['Average_Reward'])
    running_average.loc[0] = parameter_values + [reward]

print('Q-table and average reward table initialized.')

# Scale the data for use in GPR and Bayesian optimization
scaler = MinMaxScaler()
param_bounds = ([[velma_parameters[param]['min'], velma_parameters[param]['max']] for param in velma_parameters])
param_bounds = np.array(param_bounds).T
scaled_bounds = scaler.fit_transform(param_bounds)
scaled_bounds = scaled_bounds.T
param_space = [(bound[0], bound[1]) for bound in scaled_bounds]

# Gaussian Process Regression (surrogate model)
X = running_average.iloc[:, :-1].values
X_scaled = scaler.transform(X)
Y = running_average['Average_Reward'].values
gp_model = GaussianProcessRegressor(n_restarts_optimizer=20)
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
    
    # If there are less than 8 points in the q_table, force random exploration of the parameter space
    if len(q_table) < 8:
        print(f"Q-table is too sparse. Forcing random exploration of parameter space.")    
        parameter_values = [round(np.random.uniform(velma_parameters[param]['min'], velma_parameters[param]['max']), 2) 
                            for param in velma_parameters]
        for idx, param in enumerate(velma_parameters.keys()):
            velma_parameters[param]['value'] = parameter_values[idx]
        # Formats parameters so they can modify the VELMA run
        parameter_modifiers = [f'--kv="{param}",{velma_parameters[param]["value"]}' for param in velma_parameters]
        print(f"Values for {year} were changed to:")
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
        simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}/'
                                        f'Results_{outlet_id}/AnnualHydrologyResults.csv')
    else:
        simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}/AnnualHydrologyResults.csv')
    nse = simulated_results.loc[simulated_results['YEAR'] == year, 'Runoff_Nash-Sutcliffe_Coefficient']
    nse = nse.values[0]
    
    print(f"{year} had NSE of {round(nse, 2)}.")
    
    # Calculate reward
    reward = evaluate_model(year, nse)
    print(f"Reward for {year} was {round(reward, 3)}.")

    # Update the Q-table
    q_table.loc[len(q_table)] = parameter_values + [reward, nse, year]
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
            # Create a new row for the running average DataFrame
            new_row = parameter_values + [mean_reward]
            running_average.loc[len(running_average)] = new_row
            
    # Save the running_averages to a .csv
    running_average.to_csv(running_average_output, index=False)
    
    # Update surrogate model (Gaussian Process Regression)
    X = running_average.iloc[:, :-1].values
    X_scaled = scaler.transform(X)
    Y = running_average['Average_Reward'].values
    gp_model.fit(X_scaled, Y)
    
    # Plot the GPR
    X_pred = np.column_stack([np.linspace(velma_parameters[param]['min'], velma_parameters[param]['max'], 100)
        for param in velma_parameters])
    X_pred_scaled = scaler.transform(X_pred)
    Y_pred, sigma = gp_model.predict(X_pred_scaled, return_std=True)
    
    for i, param in enumerate(velma_parameters.keys()):
        plt.figure(figsize=(10, 6))
        plt.plot(X[:, i], Y, 'r.', markersize=10, label='Observed Data')
        plt.plot(X_pred[:, i], Y_pred, 'b-', label='GPR Prediction')
        plt.fill_between(X_pred[:, i].flatten(), Y_pred - 1.96 * sigma, Y_pred + 1.96 * sigma,
                        alpha=0.2, color='blue', label='95% Confidence Interval')
        plt.title(f'Gaussian Process Regression Update for {velma_parameters[param]["name"]}')
        plt.xlabel(param)
        plt.ylabel('Rewards')
        plt.legend()
        plt.grid()
        plt.savefig(f'{figure_path}/gpr_{year}_{velma_parameters[param]["name"]}')
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
    best_parameters = [round(param, 2) for param in best_parameters]
    # best_parameters = [np.clip(best_parameters[i], velma_parameters[param]['min'], velma_parameters[param]['max'])
                    #    for i, param in enumerate(velma_parameters.keys())]

    # Update VELMA parameters for the next run
    for idx, param in enumerate(velma_parameters.keys()):
        velma_parameters[param]['value'] = best_parameters[idx]
    parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]
    parameter_modifiers = [f'--kv="{param},{velma_parameters[param]["value"]}"' for param in velma_parameters]

    # Print adjusted parameters for the current year
    print(f"Updated parameters for year {year+1}:")
    for param in velma_parameters.keys():
        print(f"{velma_parameters[param]['name']}: {velma_parameters[param]['value']}")       

    # Delete unnecessary data to save space
    if year > start_learning_year+2:
        shutil.rmtree(f'{results_folder_root}/Results_{str(year-2)}')
    shutil.rmtree(f'{results_folder_root}/{xml_name}')
        
print("Calibration process completed!")
