import Xmpy as np
from wind_data.wind_data import wind

# Constants
OMEGA = 7.2921e-5  # Earth's angular velocity in rad/s
PHI = 44.74  # Latitude in degrees
NU = 1e-6  # Kinematic viscosity of water in m^2/s
DEPTHS = [1.02, 3.17, 5.46, 7.92]

# 1. Calculate Coriolis parameter
f = 2 * OMEGA * np.sin(np.deg2rad(PHI))

# 2. Calculate wind stress
def calculate_wind_stress(wind_speed):
    return 0.002 * wind_speed**2

# 3. Calculate Ekman depth
D_E = (np.pi/2) * np.sqrt(2*NU/f)

modeled_currents_with_wind = []
for data in wind:
    # Handles winds with value None
    if data['wind_speed'] is None or data['wind_dir'] is None:
        modeled_currents_with_wind.append({
        'date': data['date'],
        'time': data['time'],
        'surface_speed': None,
        'surface_dir': None,
        })
        continue
        
    wind_speed = data['wind_speed']
    wind_dir = data['wind_dir']
    
    # Calculate surface current speed and direction
    surface_speed = 0.03 * wind_speed # speed is force times mass, however I have to check if it is right calculation for water (I read somewhere that it is 3% of wind speed)
    surface_dir = wind_dir - 45  # 45° deflection from wind -> probably needs to be changed, because I have to check how ofter the angle changes
    surface_dir = surface_dir 
    
    # Calculate surface u,v components
    surface_u = surface_speed * np.cos(np.deg2rad(surface_dir))  # East-West
    surface_v = surface_speed * np.sin(np.deg2rad(surface_dir))  # North-South
    
    # Calculate currents at different depths
    depth_speeds = []
    depth_dirs = []
    depth_u = []
    depth_v = []
    
    # # CHECK CHECK CHECK!!!
    # for z in DEPTHS:
    #     # Calculate speed decay with depth
    #     decay_factor = np.exp(-z/D_E)
    #     current_speed = surface_speed * decay_factor
    #     depth_speeds.append(current_speed)
        
    #     # Calculate direction (additional 45° rotation for each Ekman depth)
    #     angle_change = (z/D_E) * 45  # Additional rotation with depth
    #     current_dir = surface_dir - angle_change
    #     current_dir = current_dir 
    #     depth_dirs.append(current_dir)
        
    #     # Calculate u,v components at this depth
    #     u = current_speed * np.cos(np.deg2rad(current_dir))  # East-West
    #     v = current_speed * np.sin(np.deg2rad(current_dir))  # North-South
    #     depth_u.append(u)
    #     depth_v.append(v)
    
    modeled_currents_with_wind.append({
        'date': data['date'],
        'time': data['time'],
        'surface_speed': surface_speed,
        'surface_dir': surface_dir,
        'surface_vector': [surface_u, surface_v],
        # 'surface_u': surface_u,
        # 'surface_v': surface_v,
        # 'depth_speeds': depth_speeds,
        # 'depth_dirs': depth_dirs,
        # 'depth_u': depth_u,
        # 'depth_v': depth_v
    })

# Save modeled currents with wind data
np.save('./created_data/modeled_currents_with_wind.npy', np.array(modeled_currents_with_wind, dtype=object))
