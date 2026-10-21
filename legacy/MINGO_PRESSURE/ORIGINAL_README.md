# Barometric Coefficient Estimation from Cosmic Ray Data

This repository provides a Python-based framework for estimating the barometric coefficient (β) from time-series measurements of atmospheric pressure and cosmic ray count rates. The code performs linear or exponential regressions in sliding time windows and outputs β values with uncertainties and R² scores. The analysis includes 2D and 3D visualizations of β across time and window length.

## Features

- Merges pressure and cosmic ray rate datasets (e.g., Oulu station).
- Applies time-windowed regression to extract β coefficients.
- Supports both linear and exponential models.
- Outputs β, uncertainty, and R² values with full diagnostics.
- Saves results to CSV and provides interactive plots (scatter and contour).
- Handles real-world data with preprocessing, filtering, and normalization.

## Usage

1. **Install dependencies**:
    ```bash
    pip install -r requirements.list
    ```

2. **Prepare data**:
    - Place `pressure_OULU.csv` and `rate_OULU.csv` in the working directory.
    - Format: CSV files with `start_date_time` and either `RPRESS` or `RUNCORR` columns, semicolon-separated.

3. **Run the analysis**:
    ```bash
    python main_script.py
    ```

4. **Outputs**:
    - `filtered_data_<start>_<end>_<step>.csv`: contains β, uncertainty, and R².
    - Interactive plots showing β and quality metrics across time and window sizes.

## Visualization

- 3D and 2D scatter plots of β vs. time and window size.
- Contour maps of β, uncertainty, and R².
- Time-sliced evolution of β for different window durations.

## Requirements

- Python 3.8+
- pandas, numpy, scipy, matplotlib, tqdm

## License

This project is licensed under the MIT License.

## Contact

Author: C. Soneira-Landín  
Email: csoneira@ucm.es
