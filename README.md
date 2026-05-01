# F1-Style-Analysis-Capstone-Project

## Overview
Formula 1 is one of the most data-driven sports in the world, yet one key factor remains difficult to measure: the driver. This project aims to determine if there is a certain way of driving the vehicle (or “driving style”) that produces better results. In this project, we analyze high-frequency telemetry data to identify and classify different driving styles by aligning laps, normalizing per driver, and clustering lap-level behavior. The results are then presented through an interactive web dashboard that allows users to explore and compare driving style patterns across drivers and tracks.

## Architecture
1) Data Processing Layer

    A Python scraper is used to ingest F1 telemetry data from: [TracingInsights.com](https://tracinginsights.com/). The data is downloaded as a JSON file and stored into a PostgreSQL database.

2) Analytics and Clustering Layer

    Principal Component Analysis (PCA) via scikit-learn reduces the dimensionality of the data and feeds it into a K-means clustering algorithm. The clustering algorithm will attempt to seperate drivers into different clusters or "driving styles." Cluster quality is evaluated using silhouette scores and the Davies-Bouldin Index. The results from the K-means clustering are stored as JSON files to be used by the web dashboard.

3) Visualization and Dashboard Layer

    A JavaScript single-page application served via nginx on an Amazon EC2 instance. The data is visualized with Chart.js and Leaflet.

## Features
The codebase is divided into three main folders, each containing code for a specific stage of the data pipeline.
- data_ingestion
  - Data Scraping (from TracingInsights.com)
  - Storing data in a PostgreSQL database
- data_analysis
  - Normalization
    - Outlier Detection (IQR)
    - Principal Component Analysis (PCA)
  - KMeans Clustering
  - Saving Clustering results as JSON files
- frontend
    - Web Front End
    - Data Visualization

## Quick Start
This project utilizes Python and several Python libraries. In the same folder as requirements.txt, run the following command to ensure that all the required libraries are installed:
###
    pip install -r requirements.txt

**Ingestion:**
1) Inside data_ingestion, run scrape.py. This will download all the F1 Telemetry data from the specified year and track and store them as JSON files.
2) Still in data_ingestion, run database_intake.py. This will attempt to read the data from your downloaded JSON files and store them into a database.

**Clustering:**

3) Inside data_analysis, run kmeans_clustering.py. You will be prompted to enter which metrics and tracks you would like to cluster on. (The clustering results will be printed to the screen and stored in JSON files within the frontend folder)


### Visual Dashboard Link: https://uark-team6-f1-visual-dashboard.site/
