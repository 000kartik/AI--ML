 **Diabetes Prediction Using SVM Classifier** using Python:

## 🛠️ How to Create This Project from Scratch

---

 Create a Project Folder

Create a folder named:


Download the dataset and place the `diabetes.csv` file inside this folder.

---

 Install Python & Required Libraries

Make sure you have Python 3.7+ installed. Then open your terminal or command prompt and install the 
necessary libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter


 Pima Indians Diabetes Dataset - Kaggle

** Import Required Libraries**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


**Load the Dataset**

df = pd.read_csv('diabetes.csv')
print(df.head())

