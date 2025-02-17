import numpy as np
import matplotlib.pyplot as plt
from po_flow_rate_data.Pontelagoscuro_Flow_Data_2023 import po_flow
from wind_data.wind_data import wind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

# Loop through each depth to create models
for depth in depths:
    # Prepare the data with AWAC data only
    X_awac = []
    y_awac = []
    for i, daily_data in enumerate(daily_currents):
        X_awac.append([
            po_flow[i+1]['flow'],
            daily_wind_averages[i]['wind_speed'],
            daily_wind_averages[i]['wind_dir'],
            # daily_data[f'awac_dir_{depth}'],
            # daily_data[f'awac_speed_{depth}']
        ])
        y_awac.append([
            daily_data[f'awac_speed_{depth}'],
            daily_data[f'awac_dir_{depth}']
        ])
    
    # Split the data into training and testing sets
    X_train_awac, X_test_awac, y_train_awac, y_test_awac = train_test_split(X_awac, y_awac, test_size=0.2, random_state=42)
    
    # Train the model
    model_awac = LinearRegression()
    model_awac.fit(X_train_awac, y_train_awac)
    
    # Store the model
    models[f'model_awac_{depth}'] = model_awac
    
    # Prepare the data with both AWAC and model data
    X_awac_model = []
    y_awac_model = []
    for i, daily_data in enumerate(daily_currents):
        X_awac_model.append([
            po_flow[i+1]['flow'],
            daily_wind_averages[i]['wind_speed'],
            daily_wind_averages[i]['wind_dir'],
            # daily_data[f'awac_dir_{depth}'],
            # daily_data[f'awac_speed_{depth}'],
            daily_data[f'model_dir_{depth}'],
            daily_data[f'model_speed_{depth}']
        ])
        y_awac_model.append([
            daily_data[f'awac_speed_{depth}'],
            daily_data[f'awac_dir_{depth}']
        ])

    # Split the data into training and testing sets
    X_train_awac_model, X_test_awac_model, y_train_awac_model, y_test_awac_model = train_test_split(X_awac_model, y_awac_model, test_size=0.2, random_state=42)
    
    # Train the model
    model_awac_model = LinearRegression()
    model_awac_model.fit(X_train_awac_model, y_train_awac_model)
    
    # Store the model
    models[f'model_awac_model_{depth}'] = model_awac_model

# Evaluate the models
for depth in depths:
    # Evaluate AWAC model
    y_pred_awac = models[f'model_awac_{depth}'].predict(X_test_awac)
    
    # Convert predictions and true values to vectors
    y_pred_awac_vectors = [
        [
            pred[0] * np.cos(np.radians(pred[1])),  # e-w component
            pred[0] * np.sin(np.radians(pred[1]))   # n-s component
        ] for pred in y_pred_awac
    ]
    
    y_test_awac_vectors = [
        [
            true[0] * np.cos(np.radians(true[1])),  # e-w component
            true[0] * np.sin(np.radians(true[1]))   # n-s component
        ] for true in y_test_awac
    ]
    
    # Calculate MAE on vector components
    mae_awac = mean_absolute_error(y_test_awac_vectors, y_pred_awac_vectors)
    rmse_awac = np.sqrt(mean_squared_error(y_test_awac_vectors, y_pred_awac_vectors))
    print(f"Depth {depth} - AWAC Model - MAE: {mae_awac}, MSE: {rmse_awac}")
    
    # Evaluate AWAC + Model data model
    y_pred_awac_model = models[f'model_awac_model_{depth}'].predict(X_test_awac_model)
    
    # Convert predictions and true values to vectors
    y_pred_awac_model_vectors = [
        [
            pred[0] * np.cos(np.radians(pred[1])),  # e-w component
            pred[0] * np.sin(np.radians(pred[1]))   # n-s component
        ] for pred in y_pred_awac_model
    ]
    
    y_test_awac_model_vectors = [
        [
            true[0] * np.cos(np.radians(true[1])),  # e-w component
            true[0] * np.sin(np.radians(true[1]))   # n-s component
        ] for true in y_test_awac_model
    ]
    
    # Calculate MAE on vector components
    mae_awac_model = mean_absolute_error(y_test_awac_model_vectors, y_pred_awac_model_vectors)
    rmse_awac_model = np.sqrt(mean_squared_error(y_test_awac_model_vectors, y_pred_awac_model_vectors))
    print(f"Depth {depth} - AWAC + Model Data Model - MAE: {mae_awac_model}, RMSE: {rmse_awac_model}")

     # For each day (x), calculate cosine similarity for predicted vs test data
    cos_sim = calculate_cosine_similarity(y_test_awac_vectors, y_pred_awac_vectors)

    for day, similarity in enumerate(cos_sim):
        cosine_similarities[depth].append({
            'date': dates[day],
            'cosine_similarity': similarity,
        })


# Plot cosine similarities
plt.figure(figsize=(15, 10))
for i, depth in enumerate(depths):
    plt.subplot(2, 2, i+1)
    cos_sims = [sim['cosine_similarity'] for sim in cosine_similarities[depth]]
    plt.plot(range(len(cos_sims)), cos_sims, marker='o')
    plt.title(f'Globina {depth}m')
    plt.ylabel('Kosinusna podobnost')
    plt.ylim(-1.25, 1.25)  # Set y-axis range to -1 to 1
    plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/cosine_similarities_predicted.png')
plt.show()

# # Naložimo podatke
# df = pd.read_csv("tvoji_podatki.csv")  # prilagodi ime datoteke

# # Definiramo vhodne in izhodne spremenljivke
# X = df[['flow', 'wind_speed', 'wind_dir']]
# y = df[['current_speed', 'current_dir']]  # Napovedujemo hitrost in smer toka

# # Razdelimo na trenirni in testni del
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Izberemo preprost Random Forest model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Napovedi
# y_pred = model.predict(X_test)

# # Ocenimo model
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# print(f"MAE: {mae}, MSE: {mse}")
