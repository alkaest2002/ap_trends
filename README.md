# Aviation Psychology Topic Trends with BERTopic

This repository contains a **reproducible pipeline** for analyzing topic trends in **aviation psychology scientific publications** using **BERTopic**.

The goal is to identify **research topics** and **temporal trends** based on paper titles, abstracts, publication sources, and publication year from the PsycArticles database.

---

## 🎯 Research Objectives

- **Identify major research topics** in aviation psychology literature
- **Track topic evolution** and temporal trends over decades
- **Discover research gaps** and emerging areas in the field
- **Ensure transparency, reproducibility**, and reviewer-friendly methodology

---

## 🗃️ Dataset Overview

**Source**: PsycArticles Database (through ProQuest)  
**Domain**: Aviation Psychology Literature  
**Time Span**: ~100 years of publications  

### Initial Dataset Schema
The dataset contains the following columns:
- **`year`**: Year of publication (temporal analysis)
- **`publication`**: Journal/source name (venue analysis) 
- **`title`**: Article title (semantic content)
- **`abstract`**: Article abstract (primary text for analysis)

### Data Quality Pipeline
```
Initial Dataset:
├── Remove incomplete records
├── Year filtering  
└── Journal quality filter
```

**Journal Selection Criteria**: Minimum 20 aviation psychology publications over 100 years or being a journal specialized in aviation psychology
- Captures the interdisciplinary nature of aviation psychology
- Includes historical foundational research
- Balances comprehensiveness with source authority

---

## 🧠 Methodological Overview

This project follows a **hybrid bibliometric–semantic approach** using BERTopic:

### 1. **Text Preprocessing**
### 2. **Semantic Representation**
### 3. **Topic Discovery with BERTopic**
   - **UMAP** for dimensionality reduction of embeddings
   - **HDBSCAN** for density-based clustering
   - **c-TF-IDF** for topic representation and keyword extraction
   - Automatic determination of optimal topic count

---

## 🔧 Technical Implementation

### Core Technologies
- **BERTopic**: Advanced topic modeling framework
- **Sentence Transformers**: Semantic text embeddings
- **UMAP**: Dimensionality reduction
- **HDBSCAN**: Clustering algorithm
- **Python ecosystem**: pandas, scikit-learn, matplotlib

---

## 📂 Author

**Dr. Pierpaolo Calanna**  
IMA Milan

---