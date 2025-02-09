import numpy as np
import matplotlib.pyplot as plt
from po_flow_rate_data.Pontelagoscuro_Flow_Data_2023 import po_flow

# Load the uo, vo, v1 and v2 values
uo = np.load('./created_data/uo.npy', allow_pickle=True)
v1 = np.load('./created_data/v1.npy', allow_pickle=True)
vo = np.load('./created_data/vo.npy', allow_pickle=True)
v2 = np.load('./created_data/v2.npy', allow_pickle=True)
v1_h = np.load('./created_data/v1_h.npy', allow_pickle=True)
v2_h = np.load('./created_data/v2_h.npy', allow_pickle=True)

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
# plt.show()

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
# plt.show()

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

mae_e = [None] * len(depths_to_plot)
mae_n = [None] * len(depths_to_plot)

# Calculate MAE for each depth
def calculate_mae_e(actual, predicted, i):
    val = np.abs(actual - predicted)
    mae_e[i] = val
    return np.mean(val)

# Calculate MAE for each depth
def calculate_mae_n(actual, predicted, i):
    val = np.abs(actual - predicted)
    mae_n[i] = val
    return np.mean(val)

mae_values_east_west = np.zeros(len(depths_to_plot))
mae_values_north_south = np.zeros(len(depths_to_plot))
for i in range(len(depths_to_plot)):
    mae_values_east_west[i] = calculate_mae_e(velocities[i]['v1'], velocities[i]['uo'], i)
    mae_values_north_south[i] = calculate_mae_n(velocities[i]['v2'], velocities[i]['vo'], i)

# print(mae_e)

# Plot absolute errors for East-West (mae_e)
plt.figure(figsize=(15, 10))
for i, depth in enumerate(depths_to_plot):
    plt.subplot(2, 2, i + 1)
    plt.plot(range(len(mae_e[i])), mae_e[i], color="blue", marker="s", markersize=4, linewidth=1)
    plt.title(f"Globina {depth}m")
    plt.xlabel("Datum")
    plt.ylabel("Absolutna napaka")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xticks(np.arange(0, len(mae_e[i]), 10), ['07/19', '07/29', '08/08', '08/18', '08/28', '09/07', '09/17', '09/27'])
plt.tight_layout()
plt.savefig('./graphs/east_west_absolute_errors.png')
# plt.show()

# Plot absolute errors for North-South (mae_n)
plt.figure(figsize=(15, 10))
for i, depth in enumerate(depths_to_plot):
    plt.subplot(2, 2, i + 1)
    plt.plot(range(len(mae_n[i])), mae_n[i], color="red", marker="s", markersize=4, linewidth=1)
    plt.title(f"Globina {depth}m")
    plt.xlabel("Datum")
    plt.ylabel("Absolutna napaka")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xticks(np.arange(0, len(mae_n[i]), 10), ['07/19', '07/29', '08/08', '08/18', '08/28', '09/07', '09/17', '09/27'])
plt.tight_layout()
plt.savefig('./graphs/north_south_absolute_errors.png')
# plt.show()

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
# plt.show()

# Extract flow values from po_flow into an array
flow_values = [day['flow'] for day in po_flow]  

correlation_flow_mae_e = np.zeros(len(depths_to_plot))
correlation_flow_mae_n = np.zeros(len(depths_to_plot))
for i in range(len(depths_to_plot)):
    correlation_flow_mae_e[i] = calculate_correlation(mae_e[i], flow_values)
    correlation_flow_mae_n[i] = calculate_correlation(mae_n[i], flow_values)

# Print calculated PCC
print("\nCorrelation between Po flow rate and absolute errors:")
print("\nEast-West component:")
for depth, corr in zip(depths_to_plot, correlation_flow_mae_e):
    print(f"Depth {depth}m: {corr:.4f}")
print("\nNorth-South component:")
for depth, corr in zip(depths_to_plot, correlation_flow_mae_n):
    print(f"Depth {depth}m: {corr:.4f}")

# Calculate Spearman rank correlation coefficient - more relevant then Pears
def calculate_spearman_correlation(x, y):
    # Convert arrays to ranks
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))
    
    d = rank_x - rank_y
    d_squared = d ** 2
    n = len(x)
    rho = 1 - (6 * np.sum(d_squared)) / (n * (n**2 - 1))
    return rho

spearman_flow_mae_e = np.zeros(len(depths_to_plot))
spearman_flow_mae_n = np.zeros(len(depths_to_plot))
for i in range(len(depths_to_plot)):
    spearman_flow_mae_e[i] = calculate_spearman_correlation(mae_e[i], flow_values)
    spearman_flow_mae_n[i] = calculate_spearman_correlation(mae_n[i], flow_values)

# Print calculated Spearman correlation coefficients
print("\nSpearman correlation between Po flow rate and absolute errors:")
print("\nEast-West component:")
for depth, corr in zip(depths_to_plot, spearman_flow_mae_e):
    print(f"Depth {depth}m: {corr:.4f}")
print("\nNorth-South component:")
for depth, corr in zip(depths_to_plot, spearman_flow_mae_n):
    print(f"Depth {depth}m: {corr:.4f}")


# From here on I am importing modeled currents with wind data and analysing it 
# Load the modeled currents with wind data
modeled_currents_with_wind = np.load('./created_data/modeled_currents_with_wind.npy', allow_pickle=True)
print(len(modeled_currents_with_wind)) # 1597
# print(modeled_currents_with_wind[-1])

# Create empty dictionary to store vectors and magnitudes for v1_h and v2_h
v_h_data = {depth: [] for depth in depths_to_plot}

# Loop through each depth and calculate vector and magnitude for each day
for i, depth in enumerate(depths_to_plot):
    for day in range(len(v1_h)):
        date = v1_h[day][0]
        v_h_vector = [v1_h[day][1][i], v2_h[day][1][i]]  
        v_h_magnitude = np.sqrt(v_h_vector[0]**2 + v_h_vector[1]**2)
        v_h_data[depth].append({
            'date': date,
            'vector': v_h_vector,
            'magnitude': v_h_magnitude
        })
print(len(v_h_data[1.02])) # 1727
# print(v_h_data[1.02][-1])

# Calculate Mean Absolute Error (MAE) for each depth
mae_e = np.zeros(len(depths_to_plot))
mae_n = np.zeros(len(depths_to_plot))

for i, depth in enumerate(depths_to_plot):
    abs_errors_e = []
    abs_errors_n = []
    for day in range(len(modeled_currents_with_wind)):
        if modeled_currents_with_wind[day]['surface_speed'] is not None:
            modeled_vector = modeled_currents_with_wind[day]['surface_vector']
            v_h_vector = v_h_data[depth][day]['vector']
            abs_errors_e.append(abs(modeled_vector[0] - v_h_vector[0]))
            abs_errors_n.append(abs(modeled_vector[1] - v_h_vector[1]))
    mae_e[i] = np.mean(abs_errors_e)
    mae_n[i] = np.mean(abs_errors_n)

# Print calculated MAE
print("\nMean Absolute Error (MAE) between modeled currents and measured currents:")
print("\nEast-West component:")
for depth, error in zip(depths_to_plot, mae_e):
    print(f"Depth {depth}m: {error:.4f}")
print("\nNorth-South component:")
for depth, error in zip(depths_to_plot, mae_n):
    print(f"Depth {depth}m: {error:.4f}")

