# Paddle OCR Ray Project

This project provides a high-performance OCR (Optical Character Recognition) service using PaddleOCR's PP-OCRv5 model, deployed with Ray Serve for scalability and performance.

## Setup

1.  **Install Dependencies:**
    Install all the required Python packages using the `requirements.txt` file. It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```
    *Note: The requirements include `paddlepaddle-gpu`. If you do not have a compatible GPU, you can replace it with `paddlepaddle` in `requirements.txt` before installation.*

2.  **Deploy the OCR Service:**
    Use Ray Serve to deploy the OCR application defined in `ray_app.py`. The deployment configuration is specified in `config.yaml`.

    ```bash
    serve deploy config.yaml
    ```
    This command will start the Ray cluster (if not already running), and deploy the OCR service. The service will run on `http://127.0.0.1:8000`. The first time you run this, it will download the OCR models from Hugging Face Hub, which may take some time.

## Testing

The project is compatible with the provided `ocr_benchmark.py` script for performance testing.

**Run the Benchmark:**
Once the service is deployed and running, you can start the benchmark test from another terminal:

```bash
python ocr_benchmark.py [OPTIONS]
```

**Example:**
To run a test with 200 requests over a 5-second window using 50 worker threads:

```bash
python ocr_benchmark.py -n 200 -t 5 -w 50
```

The benchmark script will output performance metrics such as latency percentiles and throughput.

## Project Structure

-   `ray_app.py`: Contains the Ray Serve application, including the `OCRInference` deployment class that loads the model and handles requests.
-   `config.yaml`: The configuration file for Ray Serve deployment.
-   `ocr_benchmark.py`: The benchmark script for testing the OCR service (as provided).
-   `requirements.txt`: A list of all Python dependencies for this project.
-   `README.md`: This file.