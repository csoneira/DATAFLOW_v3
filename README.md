# Unified Dataflow for miniTRASGO Charged Secondary Cosmic Ray Network

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
- **Monterrey, Mexico** (coming soon)

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

## Setup and Usage

### Prerequisites
- Python 3.8 or later
- Required libraries:
  - `pandas`
  - `numpy`
  - `requests`
  - `grafana_api`
- Access credentials for the COPERNICUS database (if using reanalysis data).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/minitrasgo-dataflow.git
   cd minitrasgo-dataflow
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
1. Configure the system by editing `config.yaml` to include station-specific details and API keys.
2. Start the dataflow system:
   ```bash
   python main.py
   ```
3. Monitor the logs for real-time updates and error handling.

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
- Cayetano Soneira (Madrid Station): [csoneira@ucm.es](mailto:csoneira@ucm.es)

