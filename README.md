# AI-Based Smart Waste Segregation System — Demo

This is a runnable **CPU-friendly demo** for an image-based waste-segregation presentation.

## What works
- Upload JPG, JPEG, PNG, or WEBP images.
- Pre-trained YOLO instance-segmentation model detects common objects.
- Displays segmentation masks and object labels.
- Lets the user confirm/correct the waste category for each detected object.
- Calculates visible pixel-area share by category.
- Exports the area report as CSV.

## Important model limitation
This demo uses `yolo11n-seg.pt`, a general pre-trained segmentation model. It is **not trained on the DWSD dataset** and does not automatically recognize the project's custom Plastic/Metal/Paper/Cloth/Glass classes. Object-to-waste category suggestions are only rough suggestions; the user must confirm them. Do not describe this as a custom-trained waste classifier.

The percentages are visible pixel-area estimates, not physical weight percentages.

## Run on Windows (VS Code)

1. Extract this ZIP.
2. Open the extracted `Smart-Waste-Segregation-Demo` folder in VS Code.
3. Open Terminal in that folder.
4. (Recommended) Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

5. Install and launch:

   ```powershell
   pip install -r requirements.txt
   streamlit run app.py
   ```

   Or double-click `run_windows.bat`.

6. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## First run
The first run downloads the small `yolo11n-seg.pt` model. Internet is required for this one-time download. Later runs can use the cached model. Inference is configured for CPU, so it may take a little longer.

## Dataset and training
The converted DWSD dataset is not included in this demo ZIP because the model has not been trained on it. The previous Colab GPU limit does not prevent running this demo locally on CPU.
