import numpy as np
import matplotlib.pyplot as plt
from po_flow_rate_data.Pontelagoscuro_Flow_Data_2023 import po_flow

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

print(dates[-1])

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

# From here on I focus on methods for vector similarities, that is why I will first rearange data in a way to have vector with e-w and n-s component and its magnitud for every day for both awac and model data
# Create empty dictionaries to store vectors and magnitudes for AWAC and model
awac_data = {depth: [] for depth in depths_to_plot}
model_data = {depth: [] for depth in depths_to_plot}

# Loop through each depth and calculate vector and magnitude for each day
for i, depth in enumerate(depths_to_plot):
    for day in range(len(dates)):
        awac_vector = [velocities[i]['uo'][day], velocities[i]['vo'][day]]
        awac_magnitude = np.sqrt(awac_vector[0]**2 + awac_vector[1]**2)
        
        model_vector = [velocities[i]['v1'][day], velocities[i]['v2'][day]]
        model_magnitude = np.sqrt(model_vector[0]**2 + model_vector[1]**2)
        
        awac_data[depth].append({
            'vector': awac_vector,
            'magnitude': awac_magnitude
        })
        
        model_data[depth].append({
            'vector': model_vector,
            'magnitude': model_magnitude
        })
# print(awac_data)

# Functions to calculate vector similarities: cosine similarity, dot product similarity and euclidean distance
def calculate_cosine_similarity(x, y):
    cos_sim = ((x['vector'][0] * y['vector'][0]) + (x['vector'][1] * y['vector'][1])) / (x['magnitude'] * y['magnitude'])
    return cos_sim 

def calculate_dot_product_similarity(x, y):
    dot_product = (x['vector'][0] * y['vector'][0]) + (x['vector'][1] * y['vector'][1])
    return dot_product

def calculate_euclidean_distance(x, y):
    distance = np.sqrt((x['vector'][0] - y['vector'][0]) ** 2 + (x['vector'][1] - y['vector'][1]) ** 2)
    return distance    
    
# Initialize a list to store cosine similarities for each depth and day
similarities = {depth: [] for depth in depths_to_plot}

# Loop through each depth and calculate cosine similarity for each day
for i, depth in enumerate(depths_to_plot):
    # For each day (x), calculate cosine similarity for uo vs v1 (east-west component) and vo vs v2 (north-south component)
    for day in range(len(dates)):
        cos_sim = calculate_cosine_similarity(awac_data[depth][day], model_data[depth][day])
        dot_prod_sim = calculate_dot_product_similarity(awac_data[depth][day], model_data[depth][day])
        euclid_dist = calculate_euclidean_distance(awac_data[depth][day], model_data[depth][day])

        # print(cos_sim)
        similarities[depth].append({
            'date': dates[day],
            'cosine_similarity': cos_sim,
            'dot_product_similarity': dot_prod_sim,
            'euclidean_distance' : euclid_dist
        })

# Plot cosine similarities
plt.figure(figsize=(15, 10))
for i, depth in enumerate(depths_to_plot):
    plt.subplot(2, 2, i+1)
    cos_sims = [sim['cosine_similarity'] for sim in similarities[depth]]
    plt.plot(range(len(cos_sims)), cos_sims, marker='o')
    plt.title(f'Globina {depth}m')
    plt.xlabel('Datum')
    plt.ylabel('Kosinusna podobnost')
    plt.xticks(np.arange(0, len(x), 10), ['07/19', '07/29', '08/08', '08/18', '08/28', '09/07', '09/17', '09/27'])
    plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/cosine_similarities.png')
plt.show()

# Plot dot product similarities
plt.figure(figsize=(15, 10))
for i, depth in enumerate(depths_to_plot):
    plt.subplot(2, 2, i+1)
    dot_sims = [sim['dot_product_similarity'] for sim in similarities[depth]]
    plt.plot(range(len(dot_sims)), dot_sims, marker='o')
    plt.title(f'Globina {depth}m')
    plt.xlabel('Datum')
    plt.ylabel('Vektorski produkt')
    plt.xticks(np.arange(0, len(x), 10), ['07/19', '07/29', '08/08', '08/18', '08/28', '09/07', '09/17', '09/27'])
    plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/dot_product_similarities.png')
plt.show()

# Plot euclidean distances
plt.figure(figsize=(15, 10))
for i, depth in enumerate(depths_to_plot):
    plt.subplot(2, 2, i+1)
    euc_dists = [sim['euclidean_distance'] for sim in similarities[depth]]
    plt.plot(range(len(euc_dists)), euc_dists, marker='o')
    plt.title(f'Globina {depth}m')
    plt.xlabel('Datum')
    plt.ylabel('Evklidska razdalja')
    plt.xticks(np.arange(0, len(x), 10), ['07/19', '07/29', '08/08', '08/18', '08/28', '09/07', '09/17', '09/27'])
    plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/euclidean_distances.png')
plt.show()

# Calculate Pearson correlation coefficient
def calculate_correlation(actual, predicted):
    return np.corrcoef(actual, predicted)[0,1]

# Extract flow values from po_flow into an array
flow_values = [day['flow'] for day in po_flow]  

# Calculate correčation for 
correlation_flow_cos = np.zeros(len(depths_to_plot))
correlation_flow_euclid = np.zeros(len(depths_to_plot))
for i in range(len(depths_to_plot)):
    cos_sim_values = [sim['cosine_similarity'] for sim in similarities[depths_to_plot[i]]]
    euclid_d_values = [sim['euclidean_distance'] for sim in similarities[depths_to_plot[i]]]
    correlation_flow_cos[i] = calculate_correlation(cos_sim_values[1:], flow_values[1:]) # skip the first one, because it is nan... there was division with 0 while calculating its value
    correlation_flow_euclid[i] = calculate_correlation(euclid_d_values, flow_values)

# Print calculated PCC
print("\nCorrelation between Po flow rate and vector similarities:")
print("\nCosine similarity and flow:")
for depth, corr in zip(depths_to_plot, correlation_flow_cos):
    print(f"Depth {depth}m: {corr:.4f}")
print("\nEuclidean distance and flow:")
for depth, corr in zip(depths_to_plot, correlation_flow_euclid):
    print(f"Depth {depth}m: {corr:.4f}")