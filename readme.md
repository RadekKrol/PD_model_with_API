# PD Default - XGBoost Model API (CSV Upload)

This project hosts a trained XGBoost model with Isotonic Calibration. The API accepts a CSV file for batch predictions.

## Setup and Installation

1.  **Clone the repository** (or download the files).

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1.  **Start the API server**:
    Once the model is saved, start the Flask API.
    ```bash
    python app.py
    ```
    The server will start on `http://127.0.0.1:5000/`.

## How to Make a Request

You can send a POST request to the `/predict` endpoint by uploading a CSV file. Make sure the CSV has a header row with column names that match the given features.

A `sample_request.csv` is provided in this project.

**Example using `curl`:**

Open a new terminal in your project directory and run the following command. This will upload the `sample_request.csv` file to the API.

```bash
curl -X POST -F "file=@sample_request.csv" http://127.0.0.1:5000/predict