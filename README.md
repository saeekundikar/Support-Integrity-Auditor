# Support Integrity Auditor

## Overview

Support Integrity Auditor is an AI-powered ticket auditing system that detects inconsistencies between customer support ticket content and assigned priority levels.

The system combines semantic understanding, keyword-based severity analysis, clustering, and transformer-based classification to identify tickets that may have been incorrectly prioritized, helping support teams reduce SLA violations and improve operational efficiency.

---

## Problem Statement

In large-scale customer support systems, ticket priorities are often assigned manually and can be inaccurate. Critical issues may be marked as low priority, while minor issues may be escalated unnecessarily.

These inconsistencies can lead to:

* Delayed resolution of high-impact incidents
* SLA breaches
* Revenue loss
* Poor customer experience
* Inefficient resource allocation

This project automatically detects priority mismatches using machine learning and natural language processing techniques.

---

## Features

* Ticket severity prediction using NLP
* Priority mismatch detection
* LoRA fine-tuned transformer model
* Semantic embedding analysis
* Keyword-based severity scoring
* Ensemble severity fusion mechanism
* Support ticket auditing and validation
* Adversarial robustness testing

---

## Tech Stack

### Machine Learning & NLP

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* Sentence Transformers

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Models

* DeBERTa-v3 Transformer
* LoRA Fine-Tuning
* K-Means Clustering
* Gradient Boosting Regressor

---

## Methodology

### 1. Data Processing

* Combined ticket subject and description
* Cleaned and standardized text
* Encoded categorical variables
* Generated semantic embeddings

### 2. Semantic Severity Analysis

Generated dense vector representations using Sentence Transformers and clustered tickets using K-Means to identify severity patterns.

### 3. Keyword-Based Severity Scoring

Designed a severity scoring engine using domain-specific indicators such as:

* System outage
* Data breach
* Payment failure
* Login failure
* Revenue impact

### 4. Severity Fusion Engine

Combined:

* Keyword Severity Score
* Embedding Severity Score
* Resolution-Time Severity Score

to produce a unified severity estimate.

### 5. Transformer-Based Integrity Auditor

Fine-tuned a LoRA-enhanced DeBERTa model to classify tickets as:

* Consistent
* Mismatch

based on the relationship between ticket content and assigned priority.

---

## Results

### Classification Performance

| Metric         | Score  |
| -------------- | ------ |
| Accuracy       | 99%    |
| Macro F1 Score | 0.99   |
| Precision      | 98–99% |
| Recall         | 98–99% |

### Class-wise Performance

| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| Consistent | 0.98      | 0.98   | 0.98     |
| Mismatch   | 0.99      | 0.99   | 0.99     |

Dashboard link :  https://support-integrity-auditor-xw62baygrwp9xqsxazrnqu.streamlit.app/#300

---

## Business Impact

The proposed system can help organizations:

* Identify incorrectly prioritized tickets automatically
* Improve SLA compliance
* Reduce incident response delays
* Improve customer satisfaction
* Optimize support team workload distribution

---

## Future Improvements

* Real-time ticket auditing
* Multi-class severity prediction
* Explainable AI severity reasoning
* Active learning feedback loops
* Dashboard integration using Power BI or Streamlit

---

## Repository Structure

```text
Support-Integrity-Auditor/
│
├── notebooks/
├── data/
├── models/
├── README.md
├── requirements.txt
└── main.py
```

---

## Author

Saee Kundikar

AI/ML | Data Science | NLP | Generative AI


