# HV Scan Analysis Script

This repository contains a Python script for analyzing High Voltage (HV) scan data for the miniTRASGO cosmic ray monitoring network.

The script downloads updated in real time CSV files, filters them based on user input, and performs various calculations and visualizations. It is designed to work with data from multiple stations and supports features like reduced field calculations, efficiency analysis, and streamer percentage tracking.

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

---

## Features

- **Data Loading and Filtering**: Loads HV scan data from CSV files and filters it based on user-specified date ranges.
- **Interactive CLI**: Provides an interactive command-line interface for user input.
- **Visualizations**: Generates plots for HV vs. efficiency, streamer percentages, pressure, temperature, and more.
- **Google Drive Integration**: Downloads data files from Google Drive if they are missing or outdated.

---

## Repository Structure

```
HV_SCAN_APP/
├── DATA/                          # Directory for storing station-specific data
│   ├── MINGO01/                   # Data for station 1
│   │   └── data_1.csv             # CSV file for station 1
│   ├── MINGO02/                   # Data for station 2
│   │   └── data_2.csv             # CSV file for station 2
│   └── ...                        # Additional stations
├── CONFIG/                        # Configuration files
│   └── drive_links.txt            # File containing Google Drive links for data
├── hv_scan_analysis.py            # Main analysis script
├── requirements.list              # Package requirements
├── README.md                      # This file
└── LICENSE                        # GNU General Public License v3.0
```

---

## Installation

### Prerequisites
- Python 3.x

### Steps
1. Clone the repository:
      ```bash
      git clone https://github.com/csoneira/HV_SCAN_APP.git
      cd HV_SCAN_APP
      ```

2. Install Python dependencies using:
      ```bash
      pip install -r requirements.list
      ```
---

## Usage


Run the script from the command line from inside the HV_SCAN_APP directory:

      python3 hv_scan_analysis.py

---

### User Inputs
- **Station Number**: Enter the station number (1-4).
- **Start and End Dates**: Specify the date range for analysis (format: `YYYY-MM-DD HH:MM:SS`).
- **Reduced Field**: Enable or disable reduced field calculations.
- **Plot Options**: Choose which plots to generate (e.g., HV vs. efficiency, streamer percentages, pressure, temperature).

### Example Workflow
1. The script will prompt for user inputs.
2. If the data file is missing or outdated, it will download it from Google Drive.
3. The script will perform calculations and generate the requested plots.

---

## Configuration

### `drive_links.txt`
This file contains Google Drive links for downloading station-specific data.

---

### Data Files
- Data files should be placed in the `DATA/MINGO0X/` directories, where `X` is the station number.
- Each file should be named `data_X.csv`, where `X` is the station number.

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---

## Contact

For questions or feedback, please contact:
- **C. Soneira-Landín**: csoneira@ucm.es
- **GitHub**: [csoneira](https://github.com/csoneira)