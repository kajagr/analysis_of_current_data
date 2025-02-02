# Project Overview

This project is focused on analyzing and visualizing ocean current data, including both measured data from an AWAC device and model data. The structure of the project is organized into several directories and Python scripts, each serving a distinct purpose in data preparation, analysis, and visualization.

## Project Structure

### Folders
- **`AWAC_S1/`**  
  Contains raw measured data from the AWAC instrument. This data serves as the baseline for comparison with the model data.

- **`created_data/`**  
  Stores processed data that has been prepared for further analysis. Data here is formatted and aligned to be compatible with analysis scripts.

- **`graphs/`**  
  Contains graphs generated using the `matplotlib` library. These include visualizations of measured and modeled data, as well as comparisons of error metrics like RMSE, MAE, and PCC.

- **`po_flow_rate_data/`**  
  Contains python file with data, related to Po flow rate in July, August and September 2023.

### Python Scripts
- **`AWAC_globine.py`**  
  Prepares the measured AWAC data for analysis. This involves cleaning, interpolating, and formatting the data to align with model data.

- **`model.py`**  
  Prepares the model data for analysis. Similar to `AWAC_globine.py`, this script ensures that the model data is clean, correctly interpolated, and formatted.

- **`analiza.py`**  
  The main script for data analysis and visualization.  
  - Generates graphs comparing measured and modeled data for different depths.  
  - Computes error metrics, including:
    - Root Mean Square Error (RMSE)
    - Mean Absolute Error (MAE)
    - Pearson Correlation Coefficient (PCC)  

  The results are saved in the `graphs/` folder for visualization and reporting.

- **`vector_similarity.py`**  
  The second main script for data analysis, focused on vector similarity between measured and modeled data. It uses cosine similarity and euclidean distance to compare the vectors, as well as correlation between Po flow rate and vector similarities.
 

## How to Use
1. **Prepare the Data:**  
   Run `AWAC_globine.py` to process the measured AWAC data and `model.py` to process the model data. 

2. **Run the Analysis:**  
   Execute `analiza.py` to compute error metrics and generate graphs. The script reads the prepared data from the `created_data/` folder and outputs results in the `graphs/` folder.

3. **Visualize Results:**  
   Open the graphs in the `graphs/` folder to visualize comparisons between measured and modeled data. Use these visualizations to interpret the error metrics and understand the performance of the model.

## Dependencies
- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`

## Key Metrics Explained
- **Root Mean Square Error (RMSE):** Measures the average magnitude of the error between the model and measured data. Lower values indicate better agreement.  
- **Mean Absolute Error (MAE):** Computes the average absolute difference between the model and measured data. Like RMSE, lower values are better.  
- **Pearson Correlation Coefficient (PCC):** Measures the linear relationship between the model and measured data, ranging from -1 to 1. Values closer to 1 indicate a strong positive correlation.

## Notes
- Depths used for analysis are limited due to missing or masked values in the model data for deeper layers.
- Ensure all scripts are executed in the correct order to avoid inconsistencies in data preparation and analysis.
