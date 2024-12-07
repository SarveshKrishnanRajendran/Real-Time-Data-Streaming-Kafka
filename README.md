
# Apache Kafka Real-Time Data Streaming Projects

## Overview
This repository contains implementations of real-time data streaming pipelines using **Apache Kafka**. These projects demonstrate the integration of **Kafka** with **PySpark**, showcasing its ability to handle high-throughput, low-latency data streams in real-world applications such as **Heart Disease Prediction** and **Sentiment Analysis**.

## Projects

### 1. Heart Disease Prediction
A machine learning pipeline was developed using **PySpark**, including preprocessing steps such as:
- **StringIndexer**
- **OneHotEncoder**
- **VectorAssembler**

#### Workflow:
1. Trained and tested a classification model to predict heart disease with an F1 score of 0.8.
2. Saved the complete pipeline for real-time predictions.
3. Set up a Kafka-based streaming pipeline to:
   - Read real-time health metrics.
   - Apply the pre-trained model to predict heart disease risk.
   - Stream results back to Kafka for continuous processing.

---

### 2. Sentiment Analysis
A real-time sentiment analysis pipeline was created using Kafka and PySpark, with the following steps:
1. Input text data streamed from Kafka.
2. Preprocessing using a pre-trained pipeline.
3. Streaming predictions to an output Kafka topic for analysis.

---

## Key Features of Apache Kafka

### Advantages:
- **Scalability:** Kafka allows horizontal scaling with minimal effort.
- **High Performance:** Capable of handling large message volumes with low latency.
- **Fault Tolerance:** Achieved through partition-level replication.
- **Flexible Integration:** Kafka Connect provides easy connections to external systems.

### Disadvantages:
- **Learning Curve:** Requires significant expertise for setup and maintenance.
- **Monitoring:** Lacks built-in tools, requiring third-party solutions.
- **Dependency:** Traditional reliance on ZooKeeper adds complexity.

---

## Repository Contents
- **[heart_kafka.py](heart_kafka.py):** Code for the heart disease prediction pipeline.
- **[heart_ml.py](heart_ml.py):** Machine learning pipeline implementation for heart disease prediction.
- **[sentiment_kafka.py](sentiment_kafka.py):** Kafka-based sentiment analysis pipeline.
- **[sentiment.py](sentiment.py):** Standalone sentiment analysis implementation.
- **[presentation.pptx](presentation.pptx):** Presentation slides detailing the architecture and implementation.
- **[Rajendran_SarveshKrishnan_report.pdf](Rajendran_SarveshKrishnan_report.pdf):** Detailed report of the projects.
- **[Rajendran_SarveshKrishnan_executive_summart.pdf](Rajendran_SarveshKrishnan_executive_summart.pdf):** Executive summary of the term project.

---

## Usage Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Set up Apache Kafka on your system.
3. Install dependencies:
   ```bash
   pip install pyspark
   ```
4. Run the desired pipeline:
   ```bash
   python heart_kafka.py
   ```

---

## Contributions
Feel free to contribute by submitting issues or pull requests. All contributions are welcome!

---

## Acknowledgments
- Project guided by **Dr. Farshid Alizadeh-Shabdiz** at Boston University.
- Leveraged Apache Kafka's distributed architecture for real-time data streaming.
