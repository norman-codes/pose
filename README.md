# Pose Classification System

This repository contains a Dockerized pose classification system using Gradio for the UI, Prometheus for metrics, and Grafana for visualization.

## File Structure

```plaintext
📁 src/
├── main.py             # Entry point for the system
├── models/             # Trained pose classification models and MoveNet models
├── utils/              # Utility scripts, including an interactive visualization and
|                         a series of notebooks used to develop the models
├── data/               # Data for testing, and preprocessed training data
├── requirements.txt    # Python dependencies for the system demo

📁 deployment/
├── Dockerfile          # Docker configuration for the pose_app
├── docker-compose.yml  # Multi-container setup for Gradio, Prometheus, and Grafana

📁 monitoring/
├── prometheus/         # Prometheus configuration
├── grafana/            # Pre-configured Grafana dashboard

📁 documentation/
├── README.md                 # Main project documentation
├── POSE_proposal.pdf         # Completed project proposal template
├── report.pdf                # Business-style project report (NOT SUBMITTED, MISSING)

📁 videos/
├── README.md     # File containing YouTube video links to system demo and data gathering
                    timelapse

.gitignore              # Ignores unnecessary files
```

## Getting Started

1. Clone the repository.
```
git clone <repo_url>
cd ProjectName
```
2. Build and run the system:
```
cd deployment
docker-compose up --build
```
3. Access services
- Gradio app: http://localhost:7860
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Dependencies
- Python 3.11
- TensorFlow Lite
- Docker and Docker Compose