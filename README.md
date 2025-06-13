# MLOPS_LAB03

This project provides a robust template for serving a machine learning model (PyTorch, image classification) via a FastAPI backend. It features an extensive, built-in monitoring and alerting system using Prometheus and Gmail.

Stack used: Prometheus + External tools for Email Alerting

## 1. Prerequisites

- **Python 3.8+**
- **pip** and **virtualenv**
- **NVIDIA GPU** with **CUDA Drivers** installed (for GPU acceleration).
- **Prometheus**: For metrics collection.

## 2. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/ht-N/MLOPS_LAB03.git
cd MLOPS_LAB03
```

### Step 2: Set Up Python Environment

It is crucial to use the specified library versions to ensure compatibility.

1.  **Create and activate a virtual environment:**

    ```bash
    conda create -n <your_venv_name> python=3.11
    conda activate <your_chosen_venv_naem>
    ```

2.  **Install PyTorch with GPU (CUDA) support:**
    The following command is for a specific CUDA version. Please check the [PyTorch website](https://pytorch.org/get-started/locally/) for the command corresponding to your system's CUDA version.

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```

3.  **Install the remaining dependencies from `requirements.txt`:**

    ```bash
    pip install -r requirements.txt
    ```

### Step 3: Configure Environment Variables

The application uses a `.env` file to manage secrets for the Gmail alerting system.

1.  Create a file named `.env` in the root of the project.
2.  Add your Gmail App Password to it. **Note**: You need to generate an "App Password" from your Google Account settings, not your regular password.

    ```env
    # .env
    GMAIL_PASS="your_google_app_password"
    ```
3.  You can configure the sender and receiver emails in the `GMAIL_ALERT_CONFIG` dictionary inside `app.py`.

### Step 4: Download Model Checkpoint

The application expects the pre-trained model checkpoint to be in the `checkpoints` directory.

1.  Create a directory named `checkpoints`.
2.  Place your `best_model_logging_demo.pth` file inside it.

## 3. Running the Application
 **Start the FastAPI Application:**
    Open a terminal, activate the virtual environment (`venv`), and run:

    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```
    The API is now running and accessible at `http://localhost:8000`.
## 4. API Endpoints

Once running, you can interact with the following endpoints:

- **`POST /predict`**: Upload an image for classification.
- **`GET /metrics`**: Prometheus scraping endpoint.
- **`GET /health`**: Basic health check.
- **`GET /api-stats`**: View real-time API performance statistics.
- **`GET /model-stats`**: View real-time model performance statistics.
- **`GET /system-info`**: View system and GPU resource usage.
- **`GET /logging-status`**: Check the status of the logging system.
- **`GET /alerts/status`**: Check the status of the alerting system and current metric values.
- **`POST /alerts/test`**: Send a test email to verify the alert configuration.
