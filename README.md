# Lung Explainer

> Status: early prototype / integration scaffold. The desktop application is not yet functional end to end.

Lung Explainer is an experimental desktop project intended to make predictions and 3D attribution maps for lung CT patches easier to inspect. It brings together a Tauri/React desktop shell and Python components derived from medical-imaging explainability work.

## What is currently implemented

The Python backend contains partial building blocks for:

- loading a 3D classifier checkpoint, including MONAI DenseNet-121 and a fallback 3D encoder architecture;
- preprocessing LUNA16-style CT volumes and candidate-centred patches;
- generating 3D Integrated Gradients explanations with Captum;
- generating black-box 3D RISE explanations with random volumetric masks;
- loading the checked-in model checkpoint and calibration metadata.

## What is not yet implemented

- `backend/app/pipeline/run_inference.py` is empty, so the model, preprocessing, and explainers are not orchestrated into a runnable inference pipeline;
- the React interface is still the default Tauri greeting screen;
- the Rust layer exposes only the template `greet` command;
- there is no connection between the desktop UI and the Python backend;
- no end-to-end test or packaged release is provided;
- `backend/requirements.txt` does not yet list every imported dependency (notably MONAI and SimpleITK).

These gaps are documented deliberately so the repository is not mistaken for a completed clinical or diagnostic application.

## Repository layout

- `backend/app/models/` — checkpoint loading and 3D model definitions;
- `backend/app/preprocessing/` — CT loading, resampling, and patch preparation;
- `backend/app/explainers/` — Integrated Gradients, RISE, and shared attribution code;
- `backend/assets/checkpoints/` — the current classifier checkpoint and calibration file;
- `src/` — React/Vite frontend scaffold;
- `src-tauri/` — Tauri 2 Rust shell.

## Frontend development

Prerequisites: Node.js, Rust stable, and the Tauri 2 platform prerequisites.

```bash
npm install
npm run tauri dev
```

This command starts only the current desktop scaffold; it does not start a working inference service.

## Intended next steps

1. implement and test the Python inference pipeline;
2. make backend dependencies reproducible;
3. define a narrow Tauri-to-Python interface;
4. replace the template UI with volume, prediction, and attribution views;
5. add end-to-end validation before distributing the application.

## Disclaimer

This repository is research-oriented software and is not a medical device. It must not be used for clinical decisions.
