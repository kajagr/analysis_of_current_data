import numpy as np
import matplotlib.pyplot as plt
from po_flow_rate_data.Pontelagoscuro_Flow_Data_2023 import po_flow
from wind_data.wind_data import wind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr
import pandas as pd


# Load the uo, vo, v1 and v2 values
uo = np.load('./created_data/uo.npy', allow_pickle=True)
v1 = np.load('./created_data/v1.npy', allow_pickle=True)
vo = np.load('./created_data/vo.npy', allow_pickle=True)
v2 = np.load('./created_data/v2.npy', allow_pickle=True)

# Function to calculate daily averages of wind speed and wind direction
def daily_average_wind(data):
    daily_averages = []
    data = data[13:] # Skip the first 13 indices because data starts with 'date': '2023-07-19', 'time': '11:00:10'
    # Handle subsequent days (24 measurements each)
    for i in range(0, len(data), 24):
        day_data = data[i:i+24]
        if day_data:
            valid_wind_speeds = [day['wind_speed'] for day in day_data if day['wind_speed'] is not None]
            valid_wind_dirs = [day['wind_dir'] for day in day_data if day['wind_dir'] is not None]
            if valid_wind_speeds and valid_wind_dirs:
                wind_speed_avg = sum(valid_wind_speeds) / len(valid_wind_speeds)
                wind_dir_avg = sum(valid_wind_dirs) / len(valid_wind_dirs)
            else:
                wind_speed_avg = None
                wind_dir_avg = None
            daily_averages.append({
                'date': day_data[0]['date'],
                'wind_speed': wind_speed_avg,
                'wind_dir': wind_dir_avg
            })
    return daily_averages

# Calculate daily averages of wind speed and wind direction
daily_wind_averages = daily_average_wind(wind)


# Define the depths to consider
depths = [1.02, 3.17, 5.46, 7.92]

# Extract dates and velocities from v1, v2, uo, and vo (starting from July 20th, 2023)
dates = v1[2:-46, 0]  # Adjusted index to start from July 20th and end with Sep 23th
velocities_uo = uo[2:-46]
velocities_vo = vo[2:-46]
velocities_v1 = v1[2:-46, 1:]
velocities_v2 = v2[2:-46, 1:]

# Prepare daily current data
daily_currents = []

# Loop through each day and depth to calculate the current speed and direction
for day in range(len(dates)):
    daily_data = {'date': dates[day]}
    for i, depth in enumerate(depths):
        # Calculate AWAC vector and magnitude
        awac_vector = [velocities_uo[day][i], velocities_vo[day][i]]
        awac_magnitude = np.sqrt(awac_vector[0]**2 + awac_vector[1]**2)
        awac_direction = np.degrees(np.arctan2(awac_vector[1], awac_vector[0]))
        
        # Calculate model vector and magnitude
        # Adjusted to handle the structure of velocities_v1 and velocities_v2
        model_vector = [velocities_v1[day][0][i], velocities_v2[day][0][i]]
        model_magnitude = np.sqrt(model_vector[0]**2 + model_vector[1]**2)
        model_direction = np.degrees(np.arctan2(model_vector[1], model_vector[0]))
        
        # Store the results
        daily_data[f'awac_speed_{depth}'] = awac_magnitude
        daily_data[f'awac_dir_{depth}'] = awac_direction
        daily_data[f'model_speed_{depth}'] = model_magnitude
        daily_data[f'model_dir_{depth}'] = model_direction
    
    daily_currents.append(daily_data)

# # Print the daily currents
# for daily_data in daily_currents:
#     print(daily_data)

# Function to calculate cosine similarity between two sets of vectors
def calculate_cosine_similarity(y_true_vectors, y_pred_vectors):
    cos_similarities = []
    for true_vec, pred_vec in zip(y_true_vectors, y_pred_vectors):
        dot_product = np.dot(true_vec, pred_vec)
        magnitude_true = np.linalg.norm(true_vec)
        magnitude_pred = np.linalg.norm(pred_vec)
        cos_sim = dot_product / (magnitude_true * magnitude_pred)
        cos_similarities.append(cos_sim)
    return cos_similarities

# Define the input and output data for the models
models = {}

# Initialize a list to store cosine similarities for each depth and day
cosine_similarities = {depth: [] for depth in depths}

# Initialize a list to store MAE and RMSE data for all depths
mae_rmse_data = []

# Initialize a list to store coefficients data for all depths
coefficients_data = []

# Loop through each depth to create models
for depth in depths:
    X_awac = []
    y_awac = []
    for i, daily_data in enumerate(daily_currents):
        X_awac.append([
            po_flow[i+1]['flow'],
            daily_wind_averages[i]['wind_speed'],
            daily_wind_averages[i]['wind_dir'],
        ])
        y_awac.append([
            daily_data[f'awac_speed_{depth}'],
            daily_data[f'awac_dir_{depth}']
        ])
    
    # Split data in time order: first 80% for training, last 20% for testing
    split_index = int(len(X_awac) * 0.8)
    X_train_awac, X_test_awac = X_awac[:split_index], X_awac[split_index:]
    y_train_awac, y_test_awac = y_awac[:split_index], y_awac[split_index:]

    model_awac = LinearRegression()
    model_awac.fit(X_train_awac, y_train_awac)

    models[f'model_awac_{depth}'] = model_awac

    # Lasso Regression for AWAC data
    lasso_awac = Lasso(alpha=0.05)
    lasso_awac.fit(X_train_awac, y_train_awac)
    models[f'lasso_awac_{depth}'] = lasso_awac

    X_awac_model = []
    y_awac_model = []
    for i, daily_data in enumerate(daily_currents):
        X_awac_model.append([
            po_flow[i+1]['flow'],
            daily_wind_averages[i]['wind_speed'],
            daily_wind_averages[i]['wind_dir'],
            daily_data[f'model_dir_{depth}'],
            daily_data[f'model_speed_{depth}']
        ])
        y_awac_model.append([
            daily_data[f'awac_speed_{depth}'],
            daily_data[f'awac_dir_{depth}']
        ])

    # Split data in time order: first 80% for training, last 20% for testing
    split_index_model = int(len(X_awac_model) * 0.8)
    X_train_awac_model, X_test_awac_model = X_awac_model[:split_index_model], X_awac_model[split_index_model:]
    y_train_awac_model, y_test_awac_model = y_awac_model[:split_index_model], y_awac_model[split_index_model:]

    model_awac_model = LinearRegression()
    model_awac_model.fit(X_train_awac_model, y_train_awac_model)
    models[f'model_awac_model_{depth}'] = model_awac_model

    # Lasso Regression for AWAC + Model data
    lasso_awac_model = Lasso(alpha=0.05)
    lasso_awac_model.fit(X_train_awac_model, y_train_awac_model)
    models[f'lasso_awac_model_{depth}'] = lasso_awac_model

# Evaluate the models
for depth in depths:
    y_pred_awac = models[f'model_awac_{depth}'].predict(X_test_awac)

    y_pred_awac_vectors = [
        [
            pred[0] * np.cos(np.radians(pred[1])), 
            pred[0] * np.sin(np.radians(pred[1]))  
        ] for pred in y_pred_awac
    ]
    
    y_test_awac_vectors = [
        [
            true[0] * np.cos(np.radians(true[1])), 
            true[0] * np.sin(np.radians(true[1]))   
        ] for true in y_test_awac
    ]
    mae_awac = mean_absolute_error(y_test_awac_vectors, y_pred_awac_vectors)
    rmse_awac = np.sqrt(mean_squared_error(y_test_awac_vectors, y_pred_awac_vectors))
    print(f"Depth {depth} - AWAC Model - MAE: {mae_awac}, MSE: {rmse_awac}")

    # Calculate MAE for E-W and N-S components separately
    y_test_awac_east_west = [vec[0] for vec in y_test_awac_vectors]
    y_pred_awac_east_west = [vec[0] for vec in y_pred_awac_vectors]
    y_test_awac_north_south = [vec[1] for vec in y_test_awac_vectors]
    y_pred_awac_north_south = [vec[1] for vec in y_pred_awac_vectors]

    mae_awac_east_west = mean_absolute_error(y_test_awac_east_west, y_pred_awac_east_west)
    mae_awac_north_south = mean_absolute_error(y_test_awac_north_south, y_pred_awac_north_south)
    print(f"Depth {depth} - AWAC Model - MAE E-W: {mae_awac_east_west}, MAE N-S: {mae_awac_north_south}")

    y_pred_awac_model = models[f'model_awac_model_{depth}'].predict(X_test_awac_model)
    y_pred_awac_model_vectors = [
        [
            pred[0] * np.cos(np.radians(pred[1])), 
            pred[0] * np.sin(np.radians(pred[1]))  
        ] for pred in y_pred_awac_model
    ]
    y_test_awac_model_vectors = [
        [
            true[0] * np.cos(np.radians(true[1])), 
            true[0] * np.sin(np.radians(true[1]))   
        ] for true in y_test_awac_model
    ]
    mae_awac_model = mean_absolute_error(y_test_awac_model_vectors, y_pred_awac_model_vectors)
    rmse_awac_model = np.sqrt(mean_squared_error(y_test_awac_model_vectors, y_pred_awac_model_vectors))
    print(f"Depth {depth} - AWAC + Model Data Model - MAE: {mae_awac_model}, RMSE: {rmse_awac_model}")

    # Calculate MAE for E-W and N-S components separately for AWAC + Model Data Model
    y_test_awac_model_east_west = [vec[0] for vec in y_test_awac_model_vectors]
    y_pred_awac_model_east_west = [vec[0] for vec in y_pred_awac_model_vectors]
    y_test_awac_model_north_south = [vec[1] for vec in y_test_awac_model_vectors]
    y_pred_awac_model_north_south = [vec[1] for vec in y_pred_awac_model_vectors]

    mae_awac_model_east_west = mean_absolute_error(y_test_awac_model_east_west, y_pred_awac_model_east_west)
    mae_awac_model_north_south = mean_absolute_error(y_test_awac_model_north_south, y_pred_awac_model_north_south)
    print(f"Depth {depth} - AWAC + Model Data Model - MAE E-W: {mae_awac_model_east_west}, MAE N-S: {mae_awac_model_north_south}")

    cos_sim = calculate_cosine_similarity(y_test_awac_vectors, y_pred_awac_vectors)

    for day, similarity in enumerate(cos_sim):
        cosine_similarities[depth].append({
            'date': dates[day],
            'cosine_similarity': similarity,
        })

    # Evaluate Lasso Regression model for AWAC data
    y_pred_lasso_awac = models[f'lasso_awac_{depth}'].predict(X_test_awac)
    y_pred_lasso_awac_vectors = [
        [
            pred[0] * np.cos(np.radians(pred[1])), 
            pred[0] * np.sin(np.radians(pred[1]))  
        ] for pred in y_pred_lasso_awac
    ]
    mae_lasso_awac = mean_absolute_error(y_test_awac_vectors, y_pred_lasso_awac_vectors)
    rmse_lasso_awac = np.sqrt(mean_squared_error(y_test_awac_vectors, y_pred_lasso_awac_vectors))
    print(f"Depth {depth} - Lasso AWAC Model - MAE: {mae_lasso_awac}, RMSE: {rmse_lasso_awac}")

    # Evaluate Lasso Regression model for AWAC + Model data
    y_pred_lasso_awac_model = models[f'lasso_awac_model_{depth}'].predict(X_test_awac_model)
    y_pred_lasso_awac_model_vectors = [
        [
            pred[0] * np.cos(np.radians(pred[1])), 
            pred[0] * np.sin(np.radians(pred[1]))  
        ] for pred in y_pred_lasso_awac_model
    ]
    mae_lasso_awac_model = mean_absolute_error(y_test_awac_model_vectors, y_pred_lasso_awac_model_vectors)
    rmse_lasso_awac_model = np.sqrt(mean_squared_error(y_test_awac_model_vectors, y_pred_lasso_awac_model_vectors))
    print(f"Depth {depth} - Lasso AWAC + Model Data Model - MAE: {mae_lasso_awac_model}, RMSE: {rmse_lasso_awac_model}")

    # Print Lasso coefficients for AWAC data
    lasso_awac_coefficients = models[f'lasso_awac_{depth}'].coef_
    print(f"Depth {depth} - Lasso Coefficients (AWAC): {lasso_awac_coefficients}")

    # Print Lasso coefficients for AWAC + Model data
    lasso_awac_model_coefficients = models[f'lasso_awac_model_{depth}'].coef_
    print(f"Depth {depth} - Lasso Coefficients (AWAC + Model): {lasso_awac_model_coefficients}")

    # Store MAE and RMSE data
    mae_rmse_data.append({
        'Depth': depth,
        'Model': 'AWAC',
        'MAE': mae_awac,
        'RMSE': rmse_awac,
        'MAE E-W': mae_awac_east_west,
        'MAE N-S': mae_awac_north_south
    })
    mae_rmse_data.append({
        'Depth': depth,
        'Model': 'AWAC + Model',
        'MAE': mae_awac_model,
        'RMSE': rmse_awac_model,
        'MAE E-W': mae_awac_model_east_west,
        'MAE N-S': mae_awac_model_north_south
    })
    mae_rmse_data.append({
        'Depth': depth,
        'Model': 'Lasso AWAC',
        'MAE': mae_lasso_awac,
        'RMSE': rmse_lasso_awac
    })
    mae_rmse_data.append({
        'Depth': depth,
        'Model': 'Lasso AWAC + Model',
        'MAE': mae_lasso_awac_model,
        'RMSE': rmse_lasso_awac_model
    })

    # Store coefficients data
    coefficients_data.append({
        'Depth': depth,
        'Model': 'Lasso AWAC',
        'Flow Coefficient': lasso_awac_coefficients[0][0],
        'Wind Speed Coefficient': lasso_awac_coefficients[0][1],
        'Wind Direction Coefficient': lasso_awac_coefficients[0][2]
    })
    coefficients_data.append({
        'Depth': depth,
        'Model': 'Lasso AWAC + Model',
        'Flow Coefficient': lasso_awac_model_coefficients[1][0],
        'Wind Speed Coefficient': lasso_awac_model_coefficients[1][1],
        'Wind Direction Coefficient': lasso_awac_model_coefficients[1][2],
        'Model Direction Coefficient': lasso_awac_model_coefficients[1][3],
        'Model Speed Coefficient': lasso_awac_model_coefficients[1][4]
    })

# Create a DataFrame to store all MAE and RMSE data
df_mae_rmse = pd.DataFrame(mae_rmse_data)

# Export the DataFrame to a single Excel file
df_mae_rmse.to_excel('./excel_data/mae_rmse_all_depths.xlsx', index=False)

# Create a DataFrame to store all coefficients data
df_coefficients = pd.DataFrame(coefficients_data)

# Export the DataFrame to a single Excel file
df_coefficients.to_excel('./excel_data/coefficients_all_depths.xlsx', index=False)

# Calculate Spearman correlation
awac_speeds = [data[f'awac_speed_{depth}'] for data in daily_currents]
awac_dir = [data[f'awac_dir_{depth}'] for data in daily_currents]
wind_speeds = [data['wind_speed'] for data in daily_wind_averages]
wind_dirs = [data['wind_dir'] for data in daily_wind_averages]
po_flows = [flow['flow'] for flow in po_flow[1:len(awac_speeds)+1]]

spearman_corr_po_flow, _ = spearmanr(po_flows, awac_speeds)
spearman_corr_wind_speed, _ = spearmanr(wind_speeds, awac_speeds)
spearman_corr_wind_dir, _ = spearmanr(wind_dirs, awac_speeds)
spearman_corr_po_flow_dir, _ = spearmanr(po_flows, awac_dir)
spearman_corr_wind_speed_dir, _ = spearmanr(wind_speeds, awac_dir)
spearman_corr_wind_dir_dir, _ = spearmanr(wind_dirs, awac_dir)

# Initialize a list to store Spearman correlation data for all depths
all_spearman_data = []

# Loop through each depth and calculate Spearman correlations
for depth in depths:
    spearman_data = {
        'Depth': depth,
        'Spearman Correlation between Po Flow and AWAC Speed': spearman_corr_po_flow,
        'Spearman Correlation between Wind Speed and AWAC Speed': spearman_corr_wind_speed,
        'Spearman Correlation between Wind Direction and AWAC Speed': spearman_corr_wind_dir,
        'Spearman Correlation between Po Flow and AWAC Dir': spearman_corr_po_flow_dir,
        'Spearman Correlation between Wind Speed and AWAC Dir': spearman_corr_wind_speed_dir,
        'Spearman Correlation between Wind Direction and AWAC Dir': spearman_corr_wind_dir_dir
    }
    all_spearman_data.append(spearman_data)

# Create a DataFrame to store all Spearman correlation data
df_spearman = pd.DataFrame(all_spearman_data)

# Export the DataFrame to a single Excel file
df_spearman.to_excel('./excel_data/spearman_correlations_all_depths.xlsx', index=False)

# Plot cosine similarities
plt.figure(figsize=(15, 10))
for i, depth in enumerate(depths):
    plt.subplot(2, 2, i+1)
    cos_sims = [sim['cosine_similarity'] for sim in cosine_similarities[depth]]
    plt.plot(range(len(cos_sims)), cos_sims, marker='o', color='royalblue')
    plt.title(f'Globina {depth}m')
    plt.ylabel('Kosinusna podobnost')
    plt.ylim(-1.25, 1.25)  # Set y-axis range to -1 to 1
    plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/cosine_similarities_predicted.png')
plt.show()