# Ensemble Methods

This folder contains comprehensive tutorials on **ensemble learning techniques**, demonstrating how combining multiple models can improve prediction accuracy and robustness compared to single models.

## 📚 Contents

### 1. `Ensemble-Methods.ipynb`
**Introduction to Ensemble Learning: Bagging, Boosting, and Voting**

A comprehensive guide to ensemble methods with practical implementations:

- **Dataset**: Student pass/fail prediction
  - Features: StudyHours, Attendance, Assignments, SleepHours
  - Target: Pass (1) or Fail (0)
  - 200 students with realistic patterns

- **Baseline Models**:
  1. **Logistic Regression**: Linear classifier (requires scaling)
  2. **Decision Tree**: Single tree (prone to overfitting)
  3. **Random Forest**: Built-in ensemble of trees

- **Ensemble Methods**:
  1. **Bagging (Bootstrap Aggregating)**:
     - Multiple trees trained on different bootstrap samples
     - Predictions averaged (voting)
     - Reduces variance, improves stability
     - Implementation: `BaggingClassifier` with 150 trees
   
  2. **Boosting (AdaBoost)**:
     - Sequential training: each model focuses on previous errors
     - Weighted combination of weak learners
     - Reduces bias, improves accuracy
     - Implementation: `AdaBoostClassifier` with 300 shallow trees
   
  3. **Voting Classifier**:
     - Combines predictions from multiple different models
     - Hard voting (majority) or soft voting (probability average)
     - Leverages strengths of different algorithms

- **Performance Comparison**:
  - Accuracy and F1 score for all models
  - Visualization of results
  - Interpretation of ensemble benefits

**Key Learning Outcomes:**
- Understand why ensembles outperform single models
- Learn bagging vs boosting differences
- Implement multiple ensemble techniques
- Compare ensemble performance with baselines
- Understand when to use each ensemble method

### 2. `Ensemble-II.ipynb`
**Advanced Ensemble Learning: Project Habits Dataset**

A focused tutorial on ensemble methods using a coding project dataset:

- **Dataset**: Student project performance
  - Features:
    - DraftsSubmitted
    - PeerReviewsGiven
    - MeetingsWithTA
    - OnTimeSubmissions
    - WeekendCodingHours
  - Target: HighGrade (binary classification)
  - 240 students

- **Models Implemented**:
  1. **Logistic Regression**: Baseline linear model
  2. **Decision Tree**: Single tree classifier
  3. **Random Forest**: Ensemble of 250 trees
  4. **Bagging**: 200 decision trees with bootstrap sampling
  5. **AdaBoost**: 300 shallow trees with adaptive boosting

- **Evaluation**:
  - Accuracy and F1 score comparison
  - Sorted results by F1 score
  - Visualization of model performance
  - Best model identification

**Key Learning Outcomes:**
- Apply ensemble methods to different domain (coding projects)
- Compare multiple ensemble techniques
- Understand feature importance in ensembles
- Practice complete ML workflow with ensembles
- Learn to select best ensemble method

## 🛠️ Technologies Used

- **pandas**: Data manipulation
- **numpy**: Numerical operations and random number generation
- **matplotlib**: Visualization
- **scikit-learn**:
  - `LogisticRegression`: Baseline linear model
  - `DecisionTreeClassifier`: Single tree
  - `RandomForestClassifier`: Tree ensemble (built-in bagging)
  - `BaggingClassifier`: Custom bagging implementation
  - `AdaBoostClassifier`: Adaptive boosting
  - `VotingClassifier`: Voting ensemble (in first notebook)
  - `StandardScaler`: Feature scaling
  - `train_test_split`: Data splitting
  - `accuracy_score`, `f1_score`, `classification_report`: Evaluation

## 📋 Prerequisites

- Understanding of classification algorithms
- Familiarity with decision trees and logistic regression
- Knowledge of train/test splits
- Understanding of bias-variance trade-off
- Basic statistics (bootstrap sampling)

## 🚀 Getting Started

1. **Install Required Packages**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

2. **Run the Notebooks**:
   - Start with `Ensemble-Methods.ipynb` for comprehensive introduction
   - Follow with `Ensemble-II.ipynb` for additional practice

## 📊 Datasets Used

1. **Student Lifestyle Dataset** (Ensemble-Methods.ipynb):
   - 200 students
   - Features: StudyHours, Attendance, Assignments, SleepHours
   - Realistic pass/fail patterns based on lifestyle factors

2. **Project Habits Dataset** (Ensemble-II.ipynb):
   - 240 students
   - Features: DraftsSubmitted, PeerReviewsGiven, MeetingsWithTA, OnTimeSubmissions, WeekendCodingHours
   - Coding project performance prediction

## 💡 Key Concepts

### Ensemble Learning Principles

**Why Ensembles Work:**
- **Diversity**: Different models make different errors
- **Averaging**: Combining predictions reduces variance
- **Robustness**: Less sensitive to outliers and noise

### Bagging (Bootstrap Aggregating)
- **Process**: 
  1. Create multiple bootstrap samples (with replacement)
  2. Train a model on each sample
  3. Average predictions (regression) or vote (classification)
- **Effect**: Reduces variance, improves stability
- **Example**: Random Forest

### Boosting
- **Process**:
  1. Train weak learner on full data
  2. Identify misclassified samples
  3. Train next learner focusing on errors
  4. Combine with weights
- **Effect**: Reduces bias, improves accuracy
- **Example**: AdaBoost, Gradient Boosting

### Voting
- **Hard Voting**: Majority class wins
- **Soft Voting**: Average probabilities, then predict
- **Effect**: Leverages different model strengths

## 🎯 Algorithm Comparison

| Method | Type | Strengths | Weaknesses | Best For |
|--------|------|-----------|------------|----------|
| **Single Tree** | Base | Interpretable, fast | Overfitting, unstable | Small datasets, interpretability |
| **Random Forest** | Bagging | Robust, handles non-linear, feature importance | Less interpretable, memory intensive | Complex patterns, large datasets |
| **Bagging** | Ensemble | Reduces variance, stable | May not reduce bias | High variance models |
| **AdaBoost** | Boosting | High accuracy, reduces bias | Sensitive to outliers, sequential | When accuracy is critical |

## 📝 Notes

- All datasets are synthetically generated with realistic patterns
- Results are reproducible with fixed random seeds
- Performance comparisons show clear ensemble benefits
- Code is well-commented for educational purposes
- Both notebooks demonstrate complete workflows

## 🔗 Related Topics

- **Classification**: See `Classification/` folder for base algorithms
- **Model Evaluation**: See `model_evaluation/` folder for metrics
- **Decision Trees**: Understanding base learners is important for ensembles

## 💡 Real-World Applications

Ensemble methods are widely used in:
- **Kaggle Competitions**: Often winners use ensemble methods
- **Production ML Systems**: Random Forest for robust predictions
- **Medical Diagnosis**: Combining multiple models for accuracy
- **Financial Modeling**: Risk assessment with ensemble predictions
- **Recommendation Systems**: Combining collaborative and content-based filtering
