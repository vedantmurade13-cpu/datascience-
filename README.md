# 📰 Fake News Detection System – Data Analysis

## 📌 Introduction

The **Fake News Detection System** is a data analysis and machine learning project designed to analyze news articles and identify whether the given news is **Real or Fake**.

In this project, the dataset is explored, cleaned, and prepared for machine learning using **Python and Jupyter Notebook**. Various data analysis and visualization techniques are used to understand the dataset, identify missing or duplicate values, analyze text data, and prepare the data for further machine learning tasks.

The main goal of this project is to create a clean and reliable dataset that can be used to train a machine learning model for detecting fake news.

---

## 🎯 Objectives

* Analyze the fake news dataset.
* Perform data cleaning and preprocessing.
* Handle missing and duplicate values.
* Explore the distribution of Real and Fake news.
* Perform exploratory data analysis (EDA).
* Visualize important patterns in the dataset.
* Prepare textual data for machine learning.
* Build a foundation for a Fake News Classification system.

---

## 🛠️ Technology Stack

### Programming Language

* **Python**

### Development Environment

* **Jupyter Notebook**

### Libraries Used

| Library          | Purpose                                   |
| ---------------- | ----------------------------------------- |
| **NumPy**        | Numerical operations and data processing  |
| **Pandas**       | Data manipulation, cleaning, and analysis |
| **Matplotlib**   | Data visualization                        |
| **Seaborn**      | Statistical data visualization            |
| **Scikit-learn** | Data preprocessing and machine learning   |
| **Regex (re)**   | Text cleaning and pattern removal         |

---

## 📊 Dataset

The project uses a dataset containing news articles and their corresponding labels.

Typical columns include:

* **Title** – Title of the news article
* **Text** – Main content of the news article
* **Subject** – Category/subject of the news
* **Date** – Date associated with the article
* **Class/Label** – Indicates whether the news is Real or Fake

### Example

| Text                    | Label |
| ----------------------- | ----- |
| News article content... | Real  |
| News article content... | Fake  |

---

## 🧹 Data Cleaning

Data cleaning is an important step because raw datasets may contain unwanted or inconsistent information.

The following operations were performed:

### 1. Checking Dataset Information

The dataset was inspected using Pandas to understand:

* Number of rows
* Number of columns
* Data types
* Missing values
* Dataset structure

```python
data.info()
```

### 2. Checking Missing Values

Missing values were identified using:

```python
data.isnull().sum()
```

This helps determine whether any columns contain empty or null values.

### 3. Checking Duplicate Records

Duplicate records were identified and removed where necessary.

```python
data.duplicated().sum()
```

### 4. Text Cleaning

The news text was cleaned before applying machine learning techniques.

Common preprocessing operations include:

* Converting text to lowercase
* Removing URLs
* Removing HTML tags
* Removing special characters
* Removing numbers
* Removing unnecessary spaces
* Removing unwanted symbols

Regular expressions were used for text cleaning.

```python
import re

text = re.sub(r'http\S+', '', text)
text = re.sub(r'<.*?>', '', text)
text = re.sub(r'[^A-Za-z\s]', '', text)
```

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the characteristics and patterns present in the dataset.

### Pandas

Pandas was mainly used for:

* Reading datasets
* Data cleaning
* Filtering data
* Handling missing values
* Checking duplicates
* Statistical analysis

```python
import pandas as pd

data = pd.read_csv("fake_news.csv")
data.head()
```

### NumPy

NumPy was used for numerical operations and working with arrays.

```python
import numpy as np
```

### Matplotlib

Matplotlib was used to create graphs and visualize the data.

```python
import matplotlib.pyplot as plt
```

### Seaborn

Seaborn was used for creating statistical visualizations.

```python
import seaborn as sns
```

Example:

```python
sns.countplot(x='class', data=data)
plt.title("Real vs Fake News")
plt.show()
```

This visualization helps understand the distribution of Real and Fake news in the dataset.

---

## 📈 Data Visualization

Different visualizations can be used to understand the dataset, including:

* Real vs Fake news distribution
* News category distribution
* Article frequency
* Text length distribution
* Word frequency
* Correlation analysis where applicable

Visualization makes it easier to identify patterns and abnormalities in the dataset.

---

## 🔤 Text Preprocessing

Since news articles are textual data, they need to be converted into a numerical format before they can be provided to a machine learning algorithm.

The text preprocessing pipeline includes:

```text
Raw News Article
        ↓
Convert to Lowercase
        ↓
Remove URLs
        ↓
Remove HTML Tags
        ↓
Remove Special Characters
        ↓
Remove Numbers
        ↓
Remove Extra Spaces
        ↓
Clean Text
        ↓
Feature Extraction
```

---

## 🤖 Machine Learning Preparation

After cleaning the text, the dataset can be divided into:

* **Input (X)** → News text
* **Output (Y)** → Real/Fake label

The dataset can then be divided into training and testing sets using `train_test_split`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
```

For converting text into numerical features, **TF-IDF Vectorization** can be used.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)
```

---

## 📁 Project Structure

```text
Fake-News-Detection/
│
├── Fake_News_Detection.ipynb
├── dataset/
│   └── fake_news.csv
│
├── README.md
│
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detection.git
```

Navigate to the project directory:

```bash
cd fake-news-detection
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Fake_News_Detection.ipynb
```

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Handling Missing/Duplicate Data
   ↓
Text Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Data Visualization
   ↓
Feature Extraction
   ↓
Train-Test Split
   ↓
Machine Learning Model
   ↓
Fake / Real Prediction
   ↓
Model Evaluation
```

---

## 📌 Key Learning Outcomes

Through this project, the following concepts were explored:

* Python programming
* Jupyter Notebook
* Pandas
* NumPy
* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Seaborn
* Matplotlib
* Natural Language Processing basics
* Text preprocessing
* TF-IDF feature extraction
* Train-test splitting
* Machine learning classification

---

## 🚀 Future Scope

The project can be further improved by:

* Implementing advanced NLP techniques.
* Comparing multiple machine learning algorithms.
* Using deep learning models.
* Adding a web-based user interface.
* Allowing users to enter news articles for prediction.
* Adding URL-based fake news detection.
* Improving model accuracy using advanced feature engineering.

---

## 👨‍💻 Author

**Vedant Murade**

### Project

**Fake News Detection System – Data Analysis & Machine Learning**

---

## ⭐ Conclusion

The Fake News Detection project demonstrates how **Python-based data analysis and machine learning techniques** can be used to analyze and classify news articles.

The project mainly focuses on **data cleaning, exploratory data analysis, visualization, and text preprocessing**, which are essential steps before building an effective machine learning model. Jupyter Notebook provides an interactive environment for performing these operations and analyzing the results.

