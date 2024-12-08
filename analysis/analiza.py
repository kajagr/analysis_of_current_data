import numpy as np
import matplotlib.pyplot as plt

# Load the uo, vo, v1 and v2 values
uo = np.load('./created_data/uo.npy', allow_pickle=True)
v1 = np.load('./created_data/v1.npy', allow_pickle=True)
vo = np.load('./created_data/vo.npy', allow_pickle=True)
v2 = np.load('./created_data/v2.npy', allow_pickle=True)

# Define the depths
depths = [1.02, 3.17, 5.46, 7.92, 10.54, 13.32, 16.27, 19.39821]

# Extract dates and velocities from v1, v2, uo and vo (skip first element because it is July 18th 2023, the day AWAC was set and -40 is because the reaserch stoped on 29th September)
dates = v1[1:-40, 0]
velocities_v1 = v1[1:-40, 1:]
velocities_v2 = v2[1:-40, 1:]
velocities_uo = uo[1:-40]
velocities_vo = vo[1:-40]

# Extract velocities for each depth
velocities = []
for i in range(4):  # For first 4 depths
    velocities.append({
        'uo': velocities_uo[:, i],
        'v1': [day[0][i] for day in velocities_v1],
        'vo': velocities_vo[:, i], 
        'v2': [day[0][i] for day in velocities_v2]
    })

x = range(len(dates))
depths_to_plot = [1.02, 3.17, 5.46, 7.92]

# Plot uo vs v1 comparison
plt.figure(figsize=(12, 8))
for i, depth in enumerate(depths_to_plot):
    plt.subplot(2, 2, i+1)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.scatter(x, velocities[i]['uo'], color="royalblue", label="Model", s=10)
    plt.scatter(x, velocities[i]['v1'], color="palevioletred", label="AWAC", s=10)
    plt.title(f"Globina: {depth}m")
    plt.xlabel("Datum")
    plt.ylabel("Hitrost toka (m/s)")
    plt.xticks(np.arange(0, len(x), 10), ['07/19', '07/29', '08/08', '08/18', '08/28', '09/07', '09/17', '09/27'])
    plt.legend()
plt.tight_layout()
plt.savefig('./graphs/uo.png')
plt.show()

# Plot vo vs v2 comparison  
plt.figure(figsize=(12, 8))
for i, depth in enumerate(depths_to_plot):
    plt.subplot(2, 2, i+1)
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.scatter(x, velocities[i]['vo'], color="royalblue", label="Model", s=10)
    plt.scatter(x, velocities[i]['v2'], color="palevioletred", label="AWAC", s=10)
    plt.title(f"Globina: {depth}m")
    plt.xlabel("Datum")
    plt.ylabel("Hitrost toka (m/s)")
    plt.xticks(np.arange(0, len(x), 10), ['07/19', '07/29', '08/08', '08/18', '08/28', '09/07', '09/17', '09/27'])
    plt.legend()
plt.tight_layout()
plt.savefig('./graphs/vo.png')
plt.show()


# Calculate RMSE for each depth
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

rmse_values_east_west = np.zeros(len(depths_to_plot))
rmse_values_north_south = np.zeros(len(depths_to_plot))
for i in range(len(depths_to_plot)):
    rmse_values_east_west[i] = calculate_rmse(velocities[i]['v1'], velocities[i]['uo'])
    rmse_values_north_south[i] = calculate_rmse(velocities[i]['v2'], velocities[i]['vo'])

# Print calculated RMSE
print("\nRoot Mean Square Error for each depth:")
print("\nEast-West component:")
for depth, rmse in zip(depths_to_plot, rmse_values_east_west):
    print(f"Depth {depth}m: {rmse:.4f}")
print("\nNorth-South component:")
for depth, rmse in zip(depths_to_plot, rmse_values_north_south):
    print(f"Depth {depth}m: {rmse:.4f}")


# Calculate MAE for each depth
def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

mae_values_east_west = np.zeros(len(depths_to_plot))
mae_values_north_south = np.zeros(len(depths_to_plot))
for i in range(len(depths_to_plot)):
    mae_values_east_west[i] = calculate_mae(velocities[i]['v1'], velocities[i]['uo'])
    mae_values_north_south[i] = calculate_mae(velocities[i]['v2'], velocities[i]['vo'])

# Print calculated MAE
print("\nMean Absolute Error for each depth:")
print("\nEast-West component:")
for depth, mae in zip(depths_to_plot, mae_values_east_west):
    print(f"Depth {depth}m: {mae:.4f}")
print("\nNorth-South component:")
for depth, mae in zip(depths_to_plot, mae_values_north_south):
    print(f"Depth {depth}m: {mae:.4f}")


# Calculate Pearson correlation coefficient for each depth
def calculate_correlation(actual, predicted):
    return np.corrcoef(actual, predicted)[0,1]

correlation_values_east_west = np.zeros(len(depths_to_plot))
correlation_values_north_south = np.zeros(len(depths_to_plot))
for i in range(len(depths_to_plot)):
    correlation_values_east_west[i] = calculate_correlation(velocities[i]['v1'], velocities[i]['uo'])
    correlation_values_north_south[i] = calculate_correlation(velocities[i]['v2'], velocities[i]['vo'])

# Print calculated PCC
print("\nPearson Correlation Coefficient for each depth:")
print("\nEast-West component:")
for depth, corr in zip(depths_to_plot, correlation_values_east_west):
    print(f"Depth {depth}m: {corr:.4f}")
print("\nNorth-South component:")
for depth, corr in zip(depths_to_plot, correlation_values_north_south):
    print(f"Depth {depth}m: {corr:.4f}")


# Create figure with 6 subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

metrics = [
    (rmse_values_east_west, rmse_values_north_south, 'RMSE', 'Kvadtatna srednja napaka'),
    (mae_values_east_west, mae_values_north_south, 'MAE', 'Srednja absolutna napaka'),
    (correlation_values_east_west, correlation_values_north_south, 'PCC', 'Pearsonov korelacijski koeficient')
]

# Visualise RMSE, MAE and PCC
for col, (east_vals, north_vals, title_base, ylabel) in enumerate(metrics):
    for row, (vals, color) in enumerate([(east_vals, 'b'), (north_vals, 'r')]):
        ax = axs[row, col]
        ax.plot(depths_to_plot, vals, f'{color}-o')
        ax.set_title(f'{title_base} - {"vzhod-zahod" if row==0 else "sever-jug"}')
        ax.set_xlabel('Globina (m)')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.set_xticks(depths_to_plot)
        ax.set_xticklabels(depths_to_plot)
plt.tight_layout()
plt.savefig('./graphs/metrics.png')
plt.show()
