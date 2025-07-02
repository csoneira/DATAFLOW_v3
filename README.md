# Unified Dataflow for miniTRASGO Cosmic Ray Network

## Overview
This repository contains the unified dataflow system for analyzing data from the **miniTRASGO** charged secondary cosmic ray network. The dataflow integrates various sources, processes them in real time, and outputs a unified dataset that can be used for visualization, monitoring, or further analysis.

The system supports:
- **Cosmic Ray (CR) Data**: Captured by miniTRASGO detectors.
- **Log Atmospheric and Performance Data**: Provides environmental and detector operational context.
- **Reanalysis Data from COPERNICUS**: Supplies high-resolution atmospheric parameters for enhanced analysis.

### Current Deployment
The network currently consists of miniTRASGO stations in the following locations:
- **Madrid, Spain**
- **Warsaw, Poland**
- **Puebla, Mexico**
- **Monterrey, Mexico**

### Key Features
- **Real-Time Data Integration**: Combines CR data, atmospheric logs, and COPERNICUS data in real-time.
- **Unified Data Table**: Generates a unified table suitable for:
  - Ingestion into a Grafana system for live visualization and monitoring.
  - Storage for future analysis.
- **Extensible Architecture**: Can be expanded to include more stations or additional data sources.

## System Architecture
The dataflow system is designed with modularity and real-time processing in mind:
1. **Data Acquisition**:
   - CR data is collected from miniTRASGO detectors.
   - Atmospheric logs and performance data are ingested from local sensors.
   - Reanalysis data is fetched from the COPERNICUS database.
2. **Data Processing**:
   - All data streams are synchronized and processed in real-time.
   - Missing data is interpolated where possible.
3. **Unified Output**:
   - A single table is generated and updated in real-time.
   - Supports both live Grafana integration and file-based storage.

## Repository Structure

### Master Tree
```
DATAFLOW_v3/MASTER
├── ZERO_STAGE
│   ├── reprocessing.sh
│   └── scheme_reprocessing_scheme.sh
├── FIRST_STAGE
│   ├── EVENT_DATA
│   │   ├── Backbone
│   │   │   ├── event_accumulator.py
│   │   │   └── raw_to_list.py
│   │   └── bring_and_analyze_events.sh
│   ├── LAB_LOGS
│   │   ├── log_aggregate_and_join.py
│   │   └── log_bring_and_clean.sh
│   └── COPERNICUS
│       └── copernicus.py
└── SECOND_STAGE
    ├── corrector.py
    └── merge_into_large_table.py
```

### Station Tree (MINGO01 Example)
```
DATAFLOW_v3/STATIONS/MINGO01
├── ZERO_STAGE
│   ├── ASCII
│   ├── COMPRESSED_HLDS
│   ├── MOVED_ASCII
│   └── UNCOMPRESSED_HLDS
├── FIRST_STAGE
│   ├── EVENT_DATA
│   │   ├── ACC_EVENTS_DIRECTORY
│   │   ├── LIST_EVENTS_DIRECTORY
│   │   ├── LIST_TO_ACC
│   │   │   └── ACC_FILES
│   │   │       ├── ACC_COMPLETED
│   │   │       ├── ACC_PROCESSING
│   │   │       └── ACC_UNPROCESSED
│   │   ├── RAW
│   │   └── RAW_TO_LIST
│   │       ├── ANCILLARY
│   │       │   ├── EMPTY_FILES
│   │       │   ├── REJECTED_FILES
│   │       │   └── TEMP_FILES
│   │       ├── PLOTS
│   │       │   ├── FIGURE_DIRECTORY
│   │       │   │   └── FIGURES_EXEC_ON_25-02-04_14.21.03
│   │       │   └── PDF_DIRECTORY
│   │       └── RAW_TO_LIST_FILES
│   │           ├── COMPLETED_DIRECTORY
│   │           ├── PROCESSING_DIRECTORY
│   │           └── UNPROCESSED_DIRECTORY
│   ├── LAB_LOGS
│   │   ├── CLEAN_LOGS
│   │   ├── LOG_ACC_DIRECTORY
│   │   ├── LOG_UNPROCESSED_DIRECTORY
│   │   └── RAW_LOGS
│   │       └── done
│   └── COPERNICUS
│       └── COPERNICUS_DATA
└── SECOND_STAGE
```

## Setup and Usage

### Prerequisites
- Python 3.8 or later
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `tqdm`
  - `Pillow`
  - `cdsapi`
  - `xarray`
- Access credentials for the COPERNICUS database (if using reanalysis data). Check the tutorial on its website.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/csoneira/DATAFLOW_v3.git
   cd DATAFLOW_v3
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.list
   ```

### Running the System
1. Configure the system by creating SSH links to every station using the `~/.ssh/config` file so the stations are called `mingo0X`, where `X` ranges from 1 to 4.
2. Start the dataflow system: add the lines in `add_to_crontab.info` to your crontab.
3. Monitor the logs for real-time updates and error handling in the tmux session, which can be set up using the text in `add_to_tmux.info`.

### Data Outputs
- **Grafana Integration**:
  - The unified table is exposed via an API endpoint that Grafana can query.
- **File-Based Storage**:
  - The unified table is periodically saved as `.csv` files in the `output/` directory.

## Contribution
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of your changes.

## Future Developments
- Deployment of the Monterrey station.
- Integration with additional reanalysis datasets.
- Automated anomaly detection in real-time CR data.

## Contact
For questions or support, please contact:
- C. Soneira-Landín (Madrid Station): [csoneira@ucm.es](mailto:csoneira@ucm.es)
