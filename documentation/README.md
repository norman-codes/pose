# Title

# Directory Structure
pose/
├── app.py                 # Main Gradio app
├── models/                # Directory for MoveNet and classification models
│   ├── movenet_thunder.tflite
│   ├── movenet_lightning.tflite
│   ├── best_both.tflite
│   ├── best_agnostic.tflite
│   ├── best_cognizant.tflite
├── Dockerfile             # Dockerfile for containerizing the app
├── requirements.txt       # Python dependencies
├── prometheus.yml         # Prometheus configuration file
├── docker-compose.yml     # Docker Compose configuration
└── README.md              # Optional: Documentation for the project

ADD LINK TO DATASET ON KAGGLE