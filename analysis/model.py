from netCDF4 import Dataset
import numpy as np

# Reading in the netCDF file
data = Dataset('cmems_mod_med_phy-cur_anfc_4.2km_P1D-m_1731243597339.nc', 'r')

# type(data)
# print(data)

# # Displaying the names of the variables
# print(data.variables.keys())

# Accessing the variables
depth = data.variables['depth']
latitude = data.variables['latitude']
longitude = data.variables['longitude']
time = data.variables['time']
uo = data.variables['uo']
vo = data.variables['vo']

# print(uo.dimensions)
# print(depth)
# print(latitude)
# print(longitude)
# print(time)
# print(uo)
# print(vo)

# Accessing the data from the variable
time_data = data.variables['time'][:]
# print(time_data)

depth_data = data.variables['depth'][:]
# print(depth_data)

latitude_data = data.variables['latitude'][:]
# print(latitude_data)

longitude_data = data.variables['longitude'][:]
# print(longitude_data)

uo_data = data.variables['uo'][:]
# print(uo_data[0][0][0])


vo_data = data.variables['vo'][:]
# print(vo_data)

def find_closest_indices(arr, value):
    if value <= arr[0]:
        return 0
    elif value >= arr[-1]:
        return 0
    else:
        for i in range(len(arr) - 1):
            if arr[i] <= value <= arr[i+1]:
                return (i, i+1)
    return None

### uo (time, depth, latitude, longitude)
### vo (time, depth, latitude, longitude)

# Longitude and latitude of AWAC
awac_lat = 44.7384
awac_lon = 12.4526

# find closest two longitude and latitude of AWAC (will use its values for interpolation)
lat1, lat2 = find_closest_indices(latitude_data, awac_lat)
lon1, lon2 = find_closest_indices(longitude_data, awac_lon)

# Create new lists of values of only all four closest longitudes and latitudes of AWAC 
uo_closest1 = uo_data[:, :, lat1, lon1]
uo_closest2 = uo_data[:, :, lat1, lon2]
uo_closest3 = uo_data[:, :, lat2, lon1]
uo_closest4 = uo_data[:, :, lat2, lon2]

vo_closest1 = vo_data[:, :, lat1, lon1]
vo_closest2 = vo_data[:, :, lat1, lon2]
vo_closest3 = vo_data[:, :, lat2, lon1]
vo_closest4 = vo_data[:, :, lat2, lon2]

# Function for bilinear interpolation
def bilinear_interpolation(x, y, x1, x2, y1, y2, q11, q12, q21, q22):
    # Calculate weights
    wx1 = (x2 - x) / (x2 - x1)
    wx2 = (x - x1) / (x2 - x1)
    wy1 = (y2 - y) / (y2 - y1)
    wy2 = (y - y1) / (y2 - y1)
    
    # Calculate interpolated value
    return (q11 * wx1 * wy1 +
            q12 * wx1 * wy2 +
            q21 * wx2 * wy1 +
            q22 * wx2 * wy2)

# Initialize array for interpolated uo values
uo_interpolated = []

# Interpolate for each timestep and depth
for t in range(len(uo_data)):
    depth_values = []
    for d in range(len(uo_data[t])):
        interpolated_value = bilinear_interpolation(
            awac_lon, awac_lat,
            longitude_data[lon1], longitude_data[lon2],
            latitude_data[lat1], latitude_data[lat2],
            uo_closest1[t][d], uo_closest2[t][d],
            uo_closest3[t][d], uo_closest4[t][d]
        )
        depth_values.append(interpolated_value)
    uo_interpolated.append(depth_values)

# Initialize array for interpolated vo values
vo_interpolated = []

# Interpolate for each timestep and depth
for t in range(len(vo_data)):
    depth_values = []
    for d in range(len(vo_data[t])):
        interpolated_value = bilinear_interpolation(
            awac_lon, awac_lat,
            longitude_data[lon1], longitude_data[lon2],
            latitude_data[lat1], latitude_data[lat2],
            vo_closest1[t][d], vo_closest2[t][d],
            vo_closest3[t][d], vo_closest4[t][d]
        )
        depth_values.append(interpolated_value)
    vo_interpolated.append(depth_values)

# Save the interpolated uo and vo values to a .npy file so we can use them for further analysis
np.save('./created_data/uo.npy', uo_interpolated)
np.save('./created_data/vo.npy', vo_interpolated)