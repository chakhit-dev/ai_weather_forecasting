# AI WEATHER FORECASTING

It's a project that uses weather data to train AI.

## Installation

```bash
  pip install requests pandas joblib scikit-learn tqdm fastapi uvicorn
```

## Usage

File `train_model_with_version` takes the provided data and trains the model, producing a `pkl` file, which is essentially the AI's brain.

```bash
py train_model_with_version.py
```

File `predict` means having the AI ​​you've trained make a prediction. The result will be displayed in a log file from the terminal.

```bash
py predict.py
```

File `main` file is used to access the API. You can enable it using a command via fastapi.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Additionally, you can download the data from the project's release page and try it out. It's currently in beta version.
