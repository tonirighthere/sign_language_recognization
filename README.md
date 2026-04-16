# Hybrid ASL Recognition

Real-time American Sign Language (ASL) Recognition Application.
Combines a lightweight CNN for static signs and a BiLSTM with Attention mechanism for dynamic signs (J & Z).

## Project Structure
- `app.py`: Main Real-time Application for Hybrid Sign Language inference through webcam.
- `notebooks/`: Jupyter Notebooks containing the End-to-End training pipeline:
  - `01_notebook.ipynb`: Data EDA & Preprocessing
  - `02_notebook.ipynb`: Bone Diagram Translation & MediaPipe Extraction
  - `03_notebook.ipynb`: Lightweight CNN Training Pipeline (Static Signs)
  - `04_notebook.ipynb`: BiLSTM Attention Training Pipeline (Dynamic Signs)
- `models/`: Required pre-trained models. (Please extract `.zip` files if present before running `app.py`)
- `requirements.txt`: Python package dependencies.

## Installation
1. Setup a virtual environment:
   ```bash
   python -m venv asl_env
   # Windows:
   .\asl_env\Scripts\activate
   # Mac/Linux:
   source asl_env/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started
Ensure that `best_cnn_model.pth`, `best_bilstm_model.pth`, and `bone_metadata.json` are placed in the `models/` directory.

Run the real-time detector:
```bash
python app.py
```

### Controls in App:
- `m`: Switch modes manually (`AUTO` -> `STATIC` -> `DYNAMIC`)
- `s`: Save a screenshot
- `q`: Quit Application
