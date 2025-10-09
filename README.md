# 🧠 CVML LAB

This repository contains a collection of **Computer Vision and Machine Learning (CVML)** lab programs, datasets, and supporting files.  
It is designed for learning and experimenting with various ML algorithms and computer vision techniques using Python.
This repository demonstrates multiple **OpenCV** and **Machine Learning** techniques using Python.
Each experiment showcases a unique concept — from basic image operations to advanced classification.1

---

## 📁 Project Structure

```

CVML LAB/
│
├── csv files/
│   ├── flowers.csv
│   ├── iris.csv
│   └── pca.csv
│
├── images/
│   ├── car.jpg
│   ├── flower.jpg
│   └── men.jpg
│
├── program1.py
├── program2.py
├── program3.py
├── program4.py
├── program5.py
├── program6.py
├── program7.py
├── program8.py
├── program9.py
│
└── README.md

````

---

## ⚙️ Requirements

Before running the programs, install the required Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn opencv-python
````

---

## 📚 Programs Overview


 📋 Summary Table

| **S.No** | **Program Name**                          | **File Name** | **Description**                                                                           | **Key Operations / Algorithms**                                    | **Libraries Used**                              | **Output / Visualization**                                       |
| :------: | :---------------------------------------- | :------------ | :---------------------------------------------------------------------------------------- | :----------------------------------------------------------------- | :---------------------------------------------- | :--------------------------------------------------------------- |
|     1    | Image Resizing, Blurring & Thresholding   | `program1.py` | Performs resizing, Gaussian blur, and binary thresholding on an image.                    | `cv2.resize()`, `cv2.GaussianBlur()`, `cv2.threshold()`            | `opencv-python`, `numpy`, `pillow`              | Displays resized, blurred & thresholded images in OpenCV windows |
|     2    | Edge Detection (Sobel, Canny, Laplacian)  | `program2.py` | Applies different edge detection algorithms to detect image boundaries.                   | `cv2.Sobel()`, `cv2.Canny()`, `cv2.Laplacian()`                    | `opencv-python`, `numpy`, `matplotlib`          | Plots Sobel X, Y, Canny, and Laplacian edges using Matplotlib    |
|     3    | Object Detection using Contours           | `program3.py` | Detects objects and draws bounding boxes using contours.                                  | `cv2.findContours()`, `cv2.rectangle()`, `cv2.putText()`           | `opencv-python`                                 | Displays detected objects with green rectangles                  |
|     4    | Feature Extraction (HOG, SIFT, ORB)       | `program4.py` | Extracts keypoints and descriptors using feature detectors.                               | `cv2.HOGDescriptor()`, `cv2.SIFT_create()`, `cv2.ORB_create()`     | `opencv-python`                                 | Displays HOG, SIFT, and ORB feature points                       |
|     5    | Face Detection using Haar Cascade         | `program5.py` | Detects faces in an image using Haar cascade classifier.                                  | `cv2.CascadeClassifier()`, `detectMultiScale()`, `cv2.rectangle()` | `opencv-python`                                 | Highlights faces with blue rectangles                            |
|     6    | Classification using SVM & KNN            | `program6.py` | Compares Support Vector Machine and K-Nearest Neighbors models.                           | `SVC()`, `KNeighborsClassifier()`, `train_test_split()`            | `pandas`, `scikit-learn`                        | Prints accuracy and classification report                        |
|     7    | Decision Tree Classification              | `program7.py` | Builds and visualizes a Decision Tree model for flower dataset.                           | `DecisionTreeClassifier()`, `plot_tree()`                          | `pandas`, `scikit-learn`, `matplotlib`          | Displays tree structure and accuracy                             |
|     8    | Logistic Regression on Iris Dataset       | `program8.py` | Performs binary classification using Logistic Regression on Iris dataset.                 | `LogisticRegression()`, `classification_report()`                  | `numpy`, `scikit-learn`                         | Prints accuracy and classification metrics                       |
|     9    | PCA & Logistic Regression on Weather Data | `program9.py` | Reduces dataset dimensionality using PCA and predicts rainfall using Logistic Regression. | `PCA()`, `StandardScaler()`, `LogisticRegression()`                | `pandas`, `numpy`, `matplotlib`, `scikit-learn` | 2D PCA scatter plot and model accuracy report                    |

---




## 🧩 Datasets

* **`flowers.csv`**, **`iris.csv`**, and **`pca.csv`** contain datasets for ML model training and testing.
* Ensure the CSV files are properly formatted and located in the `csv files/` directory.

---

## 🖼️ Images

Images such as `car.jpg`, `flower.jpg`, and `men.jpg` are used for computer vision exercises like:

* Image reading and displaying using OpenCV
* Edge detection and filtering
* Feature extraction and color space transformations

---

## 🚀 How to Run

Run any Python program using:

```bash
python program1.py
```

or open it in **Jupyter Notebook** / **VS Code** for step-by-step execution.

---

## 📊 Output Summary

| **Program** | **Output Type**    | **Visualization** |
| ----------- | ------------------ | ----------------- |
| program1.py | OpenCV Windows     | ✅ Yes             |
| program2.py | Matplotlib Plot    | ✅ Yes             |
| program3.py | Bounding Boxes     | ✅ Yes             |
| program4.py | Keypoints on Image | ✅ Yes             |
| program5.py | Face Rectangles    | ✅ Yes             |
| program6.py | Accuracy & Report  | ❌ No              |
| program7.py | Tree Visualization | ✅ Yes             |
| program8.py | Accuracy Report    | ❌ No              |
| program9.py | PCA Scatter Plot   | ✅ Yes             |

---

## 🧑‍💻 Author

**Sivasankar**
AI & ML Engineering Student

---

## 🪪 License

This project is for educational purposes.
Feel free to use or modify it for your own learning and lab work.

---
