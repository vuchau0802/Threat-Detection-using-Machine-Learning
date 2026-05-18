# Threat Detection
 
Threat Detection using Machine Learning is a FastAPI web app that detects toxic, threatening, and cyberbullying text using a hybrid machine learning approach. It combines a TF-IDF Logistic Regression model with Hugging Face Toxic-BERT to classify text as malicious or safe, provide confidence scores, severity levels, and threat categories.

The system includes prediction APIs, scan history, keyword highlighting, AWS S3 artifact storage, Docker support, CI/CD automation, and an analytics dashboard for monitoring detected threats.
 
---
 
## Features
 
- Detects toxic, threatening, cyberbullying, and unsafe text.
- Uses a hybrid model: Logistic Regression + TF-IDF + Hugging Face Toxic-BERT.
- Returns prediction label, confidence score, severity, threat category, sentiment, and highlighted keywords.
- Provides a web scanner UI and analytics dashboard.
- Stores scan history in SQLite.
- Supports model/data artifact upload and download with AWS S3.
- Supports Docker and Docker Compose for local and production-style runs.
- Includes GitHub Actions CI/CD for validation, Docker image publishing, S3 artifact upload, Hugging Face Space deployment, and AWS EC2 deployment.

## Step-by-Step Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/vuchau0802/Threat-Detection-using-Machine-Learning.git
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

### Step 3 — Activate virtual environment

```bash
.\.venv\Scripts\Activate.ps1
```

### Step 4 — Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5 — Set AWS S3 credentials
 
```powershell
$env:AWS_S3_BUCKET="threat-detection-vutrongchau"
$env:AWS_S3_KEY="threat-detection/artifacts.zip"
$env:AWS_REGION="us-east-2"
$env:AWS_ACCESS_KEY_ID="your-access-key-id"
$env:AWS_SECRET_ACCESS_KEY="your-secret-access-key"
```

### Step 6 — Install dependencies

```bash
python AWS/s3_store.py download
```

### Step 7 — Run the app

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 5000
```

### Step 8 — Open browser

```bash
http://127.0.0.1:5000
http://127.0.0.1:5000/dashboard
```

## Demo

> 

![Chat Interface](demo/4.png)

![Chat Interface](demo/1.png)

![Chat Interface](demo/2.png)

![Chat Interface](demo/3.png)

