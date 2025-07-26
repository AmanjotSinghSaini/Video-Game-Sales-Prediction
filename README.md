# Video Game Sales Prediction

This project uses machine learning models to predict whether a video game will be a commercial *hit* based on metadata like platform, genre, and publisher. The dataset used is the popular [Video Game Sales dataset](https://www.kaggle.com/datasets/gregorut/videogamesales), and the classification goal is to determine if a game has **Global Sales ≥ 1 million units**.

## Dataset Overview

The dataset contains the following columns:
- **Name, Platform, Year, Genre, Publisher**
- **Sales in NA, EU, JP, Other regions**
- **Global Sales**

The target variable is `Is_Hit`, where:
- `1` = Global Sales ≥ 1.0 million units
- `0` = Otherwise

## Workflow

1. **Data Cleaning**  
   - Dropped rows with missing values.
   - Created binary target column `Is_Hit`.

2. **Feature Encoding**  
   - Used `LabelEncoder` for categorical features: Platform, Genre, and Publisher.

3. **Feature Selection**  
   - Used `Recursive Feature Elimination (RFE)` with logistic regression to select top features.

4. **Model Training**  
   - Trained a `Random Forest Classifier` using selected features.

5. **Evaluation**  
   - Evaluated the model using accuracy, classification report, and a confusion matrix.

## Results

- **Selected Features**: `Platform`, `Genre`
- **Model Used**: RandomForestClassifier
- **Accuracy**: `~88%`

**Classification Report Highlights:**
- Class `0` (Not Hit): High precision and recall.
- Class `1` (Hit): Lower recall, indicating room for improvement in hit prediction.

## Confusion Matrix

Confusion matrix heatmap visualizes the model’s performance:
- True Negatives (TN): Correctly predicted non-hits
- False Positives (FP), etc.

## Feature Correlation

A correlation heatmap was generated to explore relationships between features and the target variable (`Is_Hit`).
