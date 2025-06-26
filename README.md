# Self-Healing Text Classification Pipeline

This project implements a **LangGraph-style DAG** pipeline for text classification using a fine-tuned transformer model. It includes a self-healing fallback mechanism when prediction confidence is low.

---

## 🔍 Objective

Build a robust CLI classifier that:
- Fine-tunes **RoBERTa-base** on the SST-2 (sentiment analysis) dataset
- Uses a LangGraph-inspired DAG:
  - `InferenceNode`: Runs prediction
  - `ConfidenceCheckNode`: Validates prediction confidence
  - `FallbackNode`: Asks user for clarification if confidence is low
- Logs all actions for analysis

---

##  Project Structure
self_healing_classifier/
- ├── cli_app.py 
- ├── dag_graph.py 
- ├── model_finetune.py 
-├── logger.py
- ├── logs/
- │ └── prediction_log.txt 
- ├── model_output/ 
- └── README.md 

## How to Run

### 1. Install dependencies
pip install transformers datasets


