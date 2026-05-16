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

## Demo

> Demo

![Chat Interface](demo/1.png)

![Chat Interface](demo/2.png)

![Chat Interface](demo/3.png)

![Chat Interface](demo/4.png)

