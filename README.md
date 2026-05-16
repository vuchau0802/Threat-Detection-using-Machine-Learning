# Threat Detection
 
An AI-powered cybersecurity threat detection application built using Flask and Machine Learning. The system analyzes security-related text inputs and predicts whether they represent malicious or safe activity.
 
---
 
## Features
 
- **Threat detection** — classifies text as Safe or Threat using a trained Logistic Regression model
- **Hard threat override** — explicit violence keywords always trigger a threat flag regardless of model score
- **Keyword highlighting** — flagged terms highlighted in the annotated output panel
- **Sentiment detection** — Positive, Negative, or Neutral per prediction
- **SQLite logging** — every prediction saved to `models/logs.db`
- **Analytics dashboard** — total scans, threat rate, confidence averages, recent logs table
- **WebSocket monitor** — real-time heartbeat connection at `/api/v1/ws/monitor`
- **REST API** — predict, logs, analytics, and health endpoints

## Step-by-Step Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/Threat-Detection-using-Machine-Learning.git
cd Threat-Detection-using-Machine-Learning
```

### Step 2 — Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download NLTK data
 
```powershell
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```
 
### Step 5 — Prepare the dataset
 
```powershell
python clean_data.py
```
 
### Step 6 — Train the model
 
```powershell
python train.py
```

### Step 7 — Run the application
 
```powershell
python app.py
```

### Step 8 — Open in browser

| http://127.0.0.1:5000/dashboard | Analytics Dashboard |

## Demo

> 

![Chat Interface](demo/4.png)

![Chat Interface](demo/1.png)

![Chat Interface](demo/2.png)

![Chat Interface](demo/3.png)

