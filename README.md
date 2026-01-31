# Mall Customers Segmentation Using K-Means Clustering

## Project Overview

This project performs **customer segmentation** on a mall dataset using **unsupervised learning** with **K-Means clustering**. By analyzing customer demographics, annual income, and spending scores, we aim to group customers into meaningful clusters. These clusters can help businesses design **targeted marketing strategies**, improve customer satisfaction, and optimize store operations.

---

## Problem Statement

Malls have diverse customers with varying purchasing behaviors. Understanding these behaviors can help businesses:

* Identify high-value customers.
* Develop personalized marketing campaigns.
* Allocate resources efficiently.

The goal of this project is to segment customers based on:

1. **Annual Income (k$)**
2. **Spending Score (1-100)**

Using these features, we group customers into clusters that share similar characteristics.

---

## Objective

* Load and preprocess the **Mall Customers dataset**.
* Perform **exploratory data analysis (EDA)** to understand patterns in customer data.
* Scale the features using **StandardScaler**.
* Determine the optimal number of clusters using:

  * **Elbow Method**
  * **Silhouette Analysis**
* Apply **K-Means clustering** to segment the customers.
* Visualize the clusters and centroids for insights.

---

##  Dataset

* **Source:** [Kaggle – Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
* **Columns:**

  * `CustomerID` – Unique customer identifier
  * `Gender` – Male or Female
  * `Age` – Age of the customer
  * `Annual Income (k$)` – Annual income of the customer in thousands
  * `Spending Score (1-100)` – Spending score assigned by the mall based on customer behavior

---

## Methodology

### 1. Import Libraries

We use standard Python libraries for data manipulation, visualization, and machine learning:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

### 2. Load and Preprocess Data

* Drop unnecessary columns like `CustomerID` and `Gender`.
* Rename columns for simplicity.
* Standardize the data using `StandardScaler`.

### 3. Exploratory Data Analysis (EDA)

* Visualize distributions for `Age`, `Annual Income`, and `Spending Score`.
* Scatterplots to explore relationships between features.

### 4. Determine Optimal Clusters

* **Elbow Method**: Plot WCSS (Within-Cluster Sum of Squares) vs number of clusters.
* **Silhouette Score**: Evaluate cluster separation quality.

### 5. Apply K-Means Clustering

* Fit K-Means with the optimal number of clusters.
* Assign cluster labels to each customer.

### 6. Visualize Results

* Scatterplots of clusters in original and scaled spaces.
* Mark centroids for better interpretation.

---

## Key Findings

* The **optimal number of clusters** is **5**, based on the elbow curve and silhouette analysis.
* Customers are segmented into:

  * Low income, low spenders
  * Low income, high spenders
  * Medium income, medium spenders
  * High income, low spenders
  * High income, high spenders
* Businesses can **target marketing campaigns** based on these clusters to improve sales and customer engagement.

---

## 🖼️ Sample Visualizations

**Cluster Plot (Annual Income vs Spending Score):**

<img width="573" height="455" alt="image" src="https://github.com/user-attachments/assets/ae364d50-2a61-4f7a-bf40-52553abdfce5" />


**Elbow Curve:**

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/68a2e30d-28c6-4e66-ab6c-b4819f07fb5f" />


**Silhouette Analysis:**

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/a0cba4cd-dc64-45de-8a50-121785e7f756" />


---

## 💻 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mall-customers-clustering.git
```

2. Navigate to the project folder:

```bash
cd mall-customers-clustering
```

3. Install required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Jupyter Notebook:

```bash
jupyter notebook Mall_Customers_Clustering.ipynb
```

---

## 📝 Requirements

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* jupyter notebook

---

## 🔗 References

* Kaggle Dataset: [Mall Customers Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
* K-Means Clustering Tutorial: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)

---

## 🏷️ License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute.

---

If you want, I can also **write a `requirements.txt` and folder structure** for this project so that it looks like a professional GitHub repo ready to share.

Do you want me to do that next?
