import datetime
import numpy as np
from datetime import timedelta

# Function that reads data from file S1_2303.sen and puts it in a list
def read_sen_file(filename):
    data_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into fields
            fields = line.split()
            # Parse the date and time
            date = datetime.datetime(int(fields[2]), int(fields[0]), int(fields[1]), 
                                     int(fields[3]), int(fields[4]), int(fields[5]))
            # Convert other fields to appropriate data types
            data = {
                'date': date,
                'status': int(fields[6], 16),
                'error': int(fields[7], 16),
                'battery': float(fields[8]),
                'soundspeed': float(fields[9]),
                'heading': float(fields[10]),
                'pitch': float(fields[11]),
                'roll': float(fields[12]),
                'pressure': float(fields[13]),
                'temperature': float(fields[14]),
                'analog1': int(fields[15]),
                'analog2': int(fields[16])
            }
            data_list.append(data)
    return data_list

# Function that reads data from file v1 and v2 and puts it in the list
def read_v_file(filename):
    data_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into fields
            fields = line.split()
            data_list.append(fields)
    return data_list

# Get v1 data
v1_data = read_v_file('AWAC_S1-GB_deployment_20230719/S1_2303.v1')
# Get v2 data
v2_data = read_v_file('AWAC_S1-GB_deployment_20230719/S1_2303.v2')
# Get sen_data
sen_data = read_sen_file('AWAC_S1-GB_deployment_20230719/S1_2303.sen')

# Create a new list of pressure values from sen_data
pressure_list = [record['pressure'] for record in sen_data]

# List of all the depths in a certain time (because of tide)
density_of_sea = 1025 # kg / m**3
gravity = 9.81 # m / s**2
depth_list = [(pressure * 10000) / (density_of_sea * gravity) for pressure in pressure_list]

# Current profile cell center distance from head (m)
distance_from_head = [ 0.90, 1.40, 1.90, 2.40, 2.90, 3.40, 3.90, 4.40, 4.90, 5.40, 5.90, 6.40, 6.90, 7.40, 7.90, 8.40, 8.90, 9.40, 9.90, 10.40, 10.90, 11.40, 11.90, 12.40, 12.90, 13.40, 13.90, 14.40, 14.90, 15.40, 15.90, 16.40, 16.90, 17.40, 17.90, 18.40, 18.90, 19.40, 19.90, 20.40, 20.90, 21.40, 21.90, 22.40, 22.90, 23.40, 23.90, 24.40, 24.90, 25.40 ]

# Create a list of lists for measured depths
measured_depths = []
for depth in depth_list:
    # Calculate measured depths for each cell
    cell_depths = [max(depth - distance, 0) for distance in distance_from_head]
    # Remove any depths that are 0 (i.e., above the water surface)
    cell_depths = [d for d in cell_depths if d > 0]
    # Add this list of depths to our main list
    measured_depths.append(cell_depths)

# Depths of model data
depths_model = [ 1.0182366,  3.1657474,  5.4649634,  7.9203773, 10.536604,  13.318384,
                 16.270586,  19.39821,   22.706392,  26.2004,    29.885643,  33.767673,
                 37.852192,  42.14504,   46.65221,   51.37986  ]

# We will use only depths from 1.0182366 to 19.39821 (later only first 4)
depth_model_7 = depths_model[:8]

# We want to invert this list because we have depths in descending order in AWAC
depth_model_7_inverted = depth_model_7[::-1]


def find_closest_indices(arr, value):
    if value >= arr[0]:
        return 0
    elif value <= arr[-1]:
        return 0
    else:
        for i in range(len(arr) - 1):
            if arr[i+1] <= value <= arr[i]:
                return (i, i+1)
    return None

# Funcion for intepolation of values to the depths of the model (in one dimension)
def interpolate(model_depth, awac_d1, awac_d2, awac_value_1, awac_value_2):
    # Calculate the weight for the interpolation
    weight = (model_depth - awac_d2) / (awac_d1 - awac_d2)
    # Perform linear interpolation
    interpolated_value = awac_value_2 + weight * (awac_value_1 - awac_value_2)
    return interpolated_value

v1_data_interpolated = []
v2_data_interpolated = []

# Interpolate every measued value to the model depth
for index, depth in enumerate(measured_depths):   
    v1_interpolated = []
    v2_interpolated = []
    if depth == []:
        v1_data_interpolated.append(v1_interpolated)
        v2_data_interpolated.append(v2_interpolated)
        continue
    for d in depth_model_7_inverted:
      i = find_closest_indices(depth, d)
      if i == 0:
        break
      value1 = interpolate(d, depth[i[0]], depth[i[1]], float(v1_data[index][i[0]]), float(v1_data[index][i[1]]))
      value2 = interpolate(d, depth[i[0]], depth[i[1]], float(v2_data[index][i[0]]), float(v2_data[index][i[1]]))
      v1_interpolated.append(value1)
      v2_interpolated.append(value2)
    v1_interpolated = v1_interpolated[::-1]  # Invert the list so depths are from 1.02m down
    v2_interpolated = v2_interpolated[::-1]  # Invert the list so depths are from 1.02m down
    v1_data_interpolated.append(v1_interpolated)
    v2_data_interpolated.append(v2_interpolated) 

# Function that handles daily averages from values mesuered every 30 minutes
def daily_average(data):
    daily_averages = []
    days = []
    start_date = datetime.datetime(2023, 7, 18, 14, 0)  # Starting from 07-18-2023 14:00
    current_date = start_date
    
    # Handle the first day separately (20 measurements)
    first_day_data = data[:20]
    if first_day_data:
        daily_averages.append([])
        days.append(current_date.date())
    
    # Handle subsequent days (48 measurements each)
    for i in range(20, len(data), 48):
        day_data = data[i:i+48]
        if day_data:
            day_sum = [0, 0, 0, 0, 0, 0, 0, 0]
            for day in day_data:
                if day == []:
                    break
                for ix in range(0,8):
                    day_sum[ix] += float(day[ix])
            daily_averages.append([sum_value / len(day_data) for sum_value in day_sum])
            current_date += timedelta(days=1)
            days.append(current_date.date())

    return list(zip(days, daily_averages))

average_daily_v1 = daily_average(v1_data_interpolated)
average_daily_v2 = daily_average(v2_data_interpolated)

# Function that handles hourly averages from values measured every 30 minutes
def hourly_average(data):
    hourly_averages = []
    hours = []
    start_date = datetime.datetime(2023, 7, 18, 14, 0)  # Starting from 07-18-2023 14:00
    current_time = start_date
    
    # Handle subsequent hours (2 measurements each)
    for i in range(2, len(data), 2):
        hour_data = data[i:i+2]
        if [] in hour_data:
            hourly_averages.append([])
            current_time += timedelta(hours=1)
            hours.append(current_time)
            continue
        if hour_data:
            hour_sum = [0, 0, 0, 0, 0, 0, 0, 0]
            for hour in hour_data:
                if hour == []:
                    break
                for ix in range(0, 8):
                    hour_sum[ix] += float(hour[ix])
            hourly_averages.append([sum_value / len(hour_data) for sum_value in hour_sum])
            current_time += timedelta(hours=1)
            hours.append(current_time)

    return list(zip(hours, hourly_averages))

# Calculate hourly averages
average_hourly_v1 = hourly_average(v1_data_interpolated)[20:-960]
average_hourly_v2 = hourly_average(v2_data_interpolated)[20:-960]

# print(average_hourly_v2[0]) #0:-960

# Convert to numpy arrays before saving (some wierd error occured)
v1_array = np.array(average_daily_v1, dtype=object)
v2_array = np.array(average_daily_v2, dtype=object)
v1_array_hourly = np.array(average_hourly_v1, dtype=object)
v2_array_hourly = np.array(average_hourly_v2, dtype=object)

# Save the modified v1 and v2 values so we can use them for further analysis (both are with daily data)
np.save('./created_data/v1.npy', v1_array)
np.save('./created_data/v2.npy', v2_array)

# Save the modified v1_array_hourly and v2_array_hourly values so we can use them for further analysis (both are with hourly data)
np.save('./created_data/v1_h.npy', v1_array_hourly)
np.save('./created_data/v2_h.npy', v2_array_hourly)