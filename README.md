# Edge AI Tactile Dashboard

This repository contains a production-ready Edge AI Dashboard for monitoring and evaluating tactile sensor data on humanoid robotics in a factory environment. The dashboard provides real-time simulation, fleet status monitoring, and historical ROI analytics.

## Prerequisites

- Python 3.10+
- A Windows environment (for running the `.bat` scripts)

## Installation

1. **Clone the repository:**
   ```cmd
   git clone https://github.com/Glaud0283/Tactile_Dashboard.git
   cd Tactile_Dashboard
   ```

2. **Create and activate a virtual environment:**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```cmd
   pip install -r Codebase/requirements.txt
   ```

## Running the Dashboard

To launch the dashboard, simply run the provided batch script:

```cmd
.\start.bat
```

This script will automatically:
1. Start the local server in the background.
2. Wait a few seconds for initialization.
3. Automatically open your default web browser to the dashboard interface (`http://127.0.0.1:8050`).

### Dashboard Features
- **Fleet Overview:** Real-time monitoring of all active sensor arrays.
- **Robot Fleet Status:** Live status and health metrics for individual robotic arms.
- **Edge AI Telemetry:** A live, animated simulation of streaming 74-channel tactile data and edge AI slippage predictions.
- **Historical Analytics:** Charts showing daily production yield efficiency.
- **AI Diagnostics:** Live performance monitoring and feature importance tracking for the deployed Random Forest model.
- **ROI & System Config:** Interactive calculators and risk mitigation matrices for factory deployment.

## Technical Details (For Data Scientists)

If you are interested in the underlying data science, exploratory data analysis (EDA), and explainable AI (XAI) feature importance that powers this dashboard, please refer to the technical notebook:

1. Open Jupyter Notebook or VS Code.
2. Navigate to `Codebase/notebooks/technical_analysis.ipynb`.
3. You can review the physical significance of the BioTac taxels and the model training procedures there.
