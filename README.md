# 🧠 Customer Segmentation using K-Means Clustering

This project performs **customer segmentation** using the **K-Means clustering algorithm**. The goal is to group customers based on their annual income and spending behavior, enabling businesses to tailor marketing strategies to distinct customer types.

## 📊 Dataset

The dataset contains the following features:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

You can find the dataset [here](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial).

## 🎯 Objectives

- Perform exploratory data analysis (EDA)
- Determine optimal number of clusters using the **Elbow Method**
- Apply **K-Means clustering** to segment customers
- Visualize customer clusters

## 🛠️ Technologies Used

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

## 📁 Project Structure

customer-segmentation-main/
├── customer_segmentation.ipynb # Main notebook with analysis and clustering
├── customer_data.csv # Dataset
├── README.md # Project documentation
└── requirements.txt # Required Python packages


## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/vritik907/customer-segmentation-main.git
   cd customer-segmentation-main

2. Install dependencies: pip install -r requirements.txt

3. Open the notebook: jupyter notebook customer_segmentation.ipynb or open file on vscode.

📈 Results
    Identified 5 distinct customer groups using K-Means.
    Clusters are visualized using scatter plots and color-coded segments.
    Results can help businesses understand customer behavior and improve targeting.

📝 License
  This project is open source and available under the MIT License.
