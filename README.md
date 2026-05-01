# F1-Style-Analysis-Capstone-Project

## Overview
Formula 1 is one of the most data-driven sports in the world, yet one key factor remains difficult to measure: **the driver**. This project aims to determine if there is a certain way of driving the vehicle (or “driving style”) that produces better results. In this project, we analyze high-frequency telemetry data to identify and classify different driving styles by aligning laps, normalizing per driver, and clustering lap-level behavior. The results are then presented through an interactive web dashboard that allows users to explore and compare driving style patterns across drivers and tracks.

## Installing Dependencies
This project utilizes Python and several Python libraries. In the same folder as requirements.txt, run the following command to ensure that all the required libraries are installed:
###
    pip install -r requirements.txt

## Features
The project is divided into three main folders, each containing code for a specific stage of the data pipeline.
- data_ingestion
  - Data Scraping (from TracingInsights.com)
  - Data Preprocessing
-data_analysis
  - Normalization
    - Outlier Detection (IQR)
    - Principal Component Analysis (PCA)
  - KMeans Clustering
  - Saving Clustering results as JSON files
-frontend
  - Web Front End
