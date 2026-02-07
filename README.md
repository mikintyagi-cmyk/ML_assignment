# Comparative Performance Evaluation of Machine Learning Classification Models



a)**Problem Statement:** This assignment focuses on the implementation and evaluation of six widely used machine learning classification models on a common dataset to ensure a fair and consistent comparison. The models considered are Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, Random Forest, and XGBoost. Model performance is evaluated using Accuracy, AUC score, Precision, Recall, F1 score, and Matthews Correlation Coefficient (MCC). The objective is to determine the most effective classification approach based on a comprehensive, metric-driven evaluation.



b) **Data set Overview:** This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) \&\& (AGI>100) \&\& (AFNLWGT>1) \&\& (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year.



**Data Set Description:** The dataset contains 32561 samples with 15 columns. Majorly age, occupation, education, relationship, capital gain etc.



**c) evaluation metrics calculated for all the 6 models**



| ML Model Name              | Accuracy | AUC    | Precision | Recall | F1     | MCC    |

|----------------------------|----------|--------|-----------|--------|--------|--------|

| Logistic Regression        | 0.7290   | 0.8357 | 0.4633    | 0.7927 | 0.5848 | 0.4357 |

| Decision Tree              | 0.7814   | 0.7044 | 0.5451    | 0.5555 | 0.5502 | 0.4508 |

| K-Nearest Neighbours       | 0.8164   | 0.8261 | 0.6748    | 0.4605 | 0.5474 | 0.4501 |

| Naive Bayes Classification | 0.7966   | 0.8290 | 0.6532    | 0.3304 | 0.4388 | 0.3592 |

| Random Forest              | 0.8075   | 0.8282 | 0.6163    | 0.5306 | 0.5703 | 0.4492 |

| XGB Classifier             | 0.8429   | 0.8733 | 0.7772    | 0.4872 | 0.5990 | 0.5920 |





d) **Observations on the Performance of Each Model:**



**Logistic Regression:** Achieves high recall, indicating effective identification of positive instances, but lower precision suggests a higher false positive rate.



**Decision Tree:** Provides balanced performance with moderate accuracy and F1 score, though its AUC indicates limited discriminative capability.



**K-Nearest Neighbors:** Demonstrates good overall accuracy and precision but comparatively lower recall, suggesting missed positive cases.



**Naive Bayes:** Performs reasonably in terms of accuracy and AUC but shows low recall, indicating weaker sensitivity to the positive class.



**Random Forest:** Offers stable and balanced performance across most metrics, with improved robustness over individual decision trees.



**XGBoost:** Delivers the best overall performance, achieving the highest accuracy, AUC, F1 score, and MCC, indicating superior predictive capability and balanced classification.

