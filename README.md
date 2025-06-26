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
├── cli_app.py # CLI interface
├── dag_graph.py # Node definitions (Inference, ConfidenceCheck, Fallback)
├── model_finetune.py # Fine-tuning RoBERTa on SST-2
├── logger.py # Logs prediction activity
├── logs/
│ └── prediction_log.txt # Auto-generated
├── model_output/ # Auto-created after fine-tuning
└── README.md # You're reading it!


## How to Run

### 1. Install dependencies
pip install transformers datasets


