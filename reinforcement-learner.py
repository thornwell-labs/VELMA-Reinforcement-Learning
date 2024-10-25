import pandas as pd
import numpy as np
import subprocess
import os
import shutil

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
velma_parallel = False
# default_results file contains starting NSE values for comparison
default_results = f'{results_folder_root}/MULTI_WS10_DefaultRun/Results_2932/AnnualHydrologyResults.csv'

# Required run parameters if velma_parallel is True
outlet_id = '2932'
max_processes = "6"

# Define starting and allowable VELMA parameter ranges
velma_parameters = {
    '/calibration/VelmaInputs.properties/be': {'value': 0.9, 'min': 0.7, 'max': 1.2},
    # '/': {'value': 1.0, 'min': 0.5, 'max': 1.5},
    # Can add/remove parameters
}


# Store values for use in reinforcement learning algorithm
parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]

# Formats parameters so they can modify the VELMA run
parameter_modifiers = [f'--kv="{param}",{velma_parameters[param]["value"]}' for param in velma_parameters]

# Run VELMA for spin-up years and output data at end to a folder
end_spinup_year = start_learning_year - 1
print(f"Running spin-up years {start_year} to {end_spinup_year}")
if velma_parallel:
    command = ["java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaParallelCmdLine", xml_path,
               f"--maxProcesses={max_processes}",
               f'--kv="/calibration/VelmaInputs.properties/eyear",{end_spinup_year}',
               f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName",'
               f'{results_folder_root}/Results_{end_spinup_year}',
               *parameter_modifiers]
    command_str = ' '.join(command)
    subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    command = ["java", allocated_memory, "-cp", jar_path, "gov.epa.velmasimulator.VelmaSimulatorCmdLine", xml_path,
               f'--kv="/calibration/VelmaInputs.properties/eyear",{end_spinup_year}',
               f'--kv="/startups/VelmaStartups.properties/setEndStateSpatialDataLocationName",'
               f'{results_folder_root}/Results_{end_spinup_year}',
               *parameter_modifiers]
    command_str = ' '.join(command)
    subprocess.run(command_str, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Initiate NSE and error using AnnualHydrologyResults
if velma_parallel:
    simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}/'
                                    f'/Results_{outlet_id}/AnnualHydrologyResults.csv')
else:
    simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}'
                                    f'/AnnualHydrologyResults.csv')
nse = simulated_results.loc[simulated_results['YEAR'] == end_spinup_year, 'Runoff_Nash-Sutcliffe_Coefficient']
nse = nse.values[0]
error = 1 - nse

print('Spin-up years complete.')
print(f'End spin-up year had NSE = {nse}.')

# Calculate the reward by comparing the new performance to the default performance
default_df = pd.read_csv(default_results, usecols=['YEAR', 'Runoff_Nash-Sutcliffe_Coefficient'])
default_nse = default_df.loc[default_df['YEAR'] == end_spinup_year, 'Runoff_Nash-Sutcliffe_Coefficient'].values[0]
reward = nse - default_nse

# Initialize Q-table
q_table = pd.DataFrame(columns=list(velma_parameters.keys())+['Reward', 'NSE', 'Error', 'Year'])
q_table.loc[len(q_table)] = parameter_values + [reward, nse, error, end_spinup_year]
# Check whether the file exists so that old data isn't overwritten
if not os.path.isfile(q_table_output):
    q_table.to_csv(q_table_output, mode='w', header=True)
else:
    q_table.to_csv(q_table_output, mode='a', header=False)

# Initialize policy (starting values), learning rate (alpha), and exploration rate (epsilon)
# Alpha and epsilon are testable hyperparameters
policy = {param: velma_parameters[param]['value'] for param in velma_parameters}
alpha = 0.2
epsilon = 0.4  # will explore new random parameters in this percentage of runs, expressed as decimal

# Check if running_averages.csv exists; if not, need to initialize it
# Initialize table of parameters to record unique values and average NSE
if os.path.exists(running_average_output):
    running_average = pd.read_csv(running_average_output)
else:
    running_average = pd.DataFrame(columns=list(velma_parameters.keys())+['Average_Reward'])
    running_average.loc[0] = parameter_values + [reward]

print('Q-table and policy initialized.')

# Remove unnecessary results
shutil.rmtree(f'{results_folder_root}/{xml_name}')

# Run the below code in a loop for the rest of the years
for year in range(start_learning_year, end_year+1):
    print(f"Running VELMA for year: {year}")

    # Locations of folders for start data and end data
    start_data = f'{results_folder_root}/Results_{str(year - 1)}'
    end_data = f'{results_folder_root}/Results_{str(year)}'

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
        

    # Find NSE and error using AnnualHydrologyResults
    if velma_parallel:
        simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}/'
                                        f'Results_{outlet_id}/AnnualHydrologyResults.csv')
    else:
        simulated_results = pd.read_csv(f'{results_folder_root}/{xml_name}/AnnualHydrologyResults.csv')
    nse = simulated_results.loc[simulated_results['YEAR'] == year, 'Runoff_Nash-Sutcliffe_Coefficient']
    nse = nse.values[0]
    error = 1 - nse
    
    print(f"{year} had NSE of {nse}.")
    
    # Calculate reward
    default_nse = default_df.loc[default_df['YEAR'] == end_spinup_year, 'Runoff_Nash-Sutcliffe_Coefficient'].values[0]
    reward = nse - default_nse

    # Update the Q-table
    q_table.loc[len(q_table)] = parameter_values + [reward, nse, error, year]
    q_table.loc[[len(q_table)-1]].to_csv(q_table_output, mode='a', header=False)
    
    # Update the table of running average NSE
    # First find matching rows in the q_table
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
    
    # Update the policy using the best performance according to the average reward
    # Random number generation to determine whether to explore or exploit
    explore_or_exploit = np.random.rand()
    if explore_or_exploit > epsilon:
        print("Chose exploitation.")
        # Exploitation: Find the best-performing parameter set and step towards it from the current parameters using the learning rate
        best_row = running_average.loc[running_average['Average_Reward'].idxmax()]
        for param in policy:
            policy[param] += round(alpha * (best_row[param] - policy[param])/best_row[param], 2)
            # Below line makes sure the new parameters stay within the given values
            policy[param] = round(np.clip(policy[param], a_min=velma_parameters[param]['min'], a_max=velma_parameters[param]['max']), 2)

    else:
        print("Chose exploration.")
        # Exploration: randomly choose a number within the parameter range
        for param in policy:
            policy[param] = round(np.random.uniform(velma_parameters[param]['min'], velma_parameters[param]['max']), 2)

    # Update VELMA parameters for the next run
    for param in velma_parameters.keys():
        velma_parameters[param]['value'] = policy[param]
    parameter_values = [velma_parameters[param]["value"] for param in velma_parameters]
    parameter_modifiers = [f'--kv="{param},{velma_parameters[param]["value"]}"' for param in velma_parameters]

    # Print adjusted parameters for the current year
    print(f"Updated parameters for year {year}:")
    for param in policy:
        print(f"{param}: {policy[param]}")          

    # Delete unnecessary data to save space
    if year > start_learning_year+2:
        shutil.rmtree(f'{results_folder_root}/Results_{str(year-2)}')
    if year != end_year:
        shutil.rmtree(f'{results_folder_root}/{xml_name}')

print("Calibration process completed!")
