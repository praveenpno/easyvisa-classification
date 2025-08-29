# EasyVisa: Visa Application Certification Prediction

## Overview
This project addresses the increasing volume of foreign worker visa applications processed by the US Office of Foreign Labor Certification (OFLC). With nearly 1.7 million positions processed in FY 2016 and an annual nine percent increase in applications, manual review is becoming unsustainable. The objective is to develop a machine learning classification model to predict visa certification status, thereby streamlining the approval process for the OFLC and identifying key factors that significantly influence the decision to certify or deny a visa.

## Technologies & Libraries
The following key technologies and Python libraries were utilized in this project:

*   **Python**
*   **Jupyter Notebook**
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib**: For data visualization.
*   **Seaborn**: For enhanced statistical data visualization.
*   **Scikit-learn**: For machine learning model building, evaluation, and data preprocessing (e.g., `train_test_split`, `DecisionTreeClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`, `BaggingClassifier`, `GridSearchCV`, `RandomizedSearchCV`, `metrics`).
*   **XGBoost**: For Extreme Gradient Boosting classification (`XGBClassifier`).
*   **imbalanced-learn (imblearn)**: For handling class imbalance (`SMOTE` for oversampling, `RandomUnderSampler` for undersampling).

## Setup & Installation
To run this project, install the required dependencies:

```bash
pip install -r requirements.txt
```

And a suggested `requirements.txt`:
```
numpy==1.25.2
pandas==1.5.3
scikit-learn==1.5.2
matplotlib==3.7.1
seaborn==0.13.1
xgboost==2.0.3
imbalanced-learn
```

## How to Run
Open the `notebook.ipynb` file in a Jupyter environment (like Jupyter Lab or VS Code) and execute the cells sequentially. Ensure you have the dataset `EasyVisa.csv` available, ideally mounted in a similar path to `/content/drive/MyDrive/AIML_Apr_2025/Projects/Project3: EasyVisa/EasyVisa.csv` if using Google Colab.

## Key Findings & Conclusion

The analysis and modeling revealed significant insights into the factors influencing visa application certification.

**Key Findings:**
*   **Applicant Profile**: The majority of applicants are from Asia, hold a Bachelor's degree, possess prior job experience, and are applying for full-time positions with yearly wages, primarily targeting the Northeast, South, and West regions of the US. Approximately 67% of applications are certified, while 33% are denied.
*   **Influential Factors for Certification**:
    *   **Education**: Applicants with Master's and Doctorate degrees exhibit a higher likelihood of visa certification.
    *   **Continent**: European applicants show the highest visa certification rates.
    *   **Job Experience**: Prior work experience significantly increases the chances of approval.
    *   **Prevailing Wage**: A positive correlation exists between higher prevailing wages and visa certification, though wages vary across regions (e.g., Midwest having a higher median).
    *   **Unit of Wage**: Applications with a yearly wage unit have a much higher approval rate compared to hourly, weekly, or monthly units.
*   **Data Quality**: Initial data checks revealed negative values in `no_of_employees` (corrected) and outliers in `prevailing_wage` and `yr_of_estab`, which represent unusual but potentially valid cases (e.g., C-level executive wages or very old companies).
*   **Model Performance**: Due to class imbalance, models trained on oversampled data (using SMOTE) consistently outperformed those on undersampled or original data, with F1-score as the primary evaluation metric (balancing False Positives and False Negatives). Among the ensemble models, the **Tuned XGBoost Classifier with Oversampled data** emerged as the best performer, achieving a validation F1-score of 0.817 and a test F1-score of 0.808.
*   **Feature Importance**: The most significant features driving visa certification predictions were the employee's education level, the region of employment, and the unit of wage.

**Conclusion & Recommendations:**
The developed Machine Learning model can effectively assist the OFLC and EasyVisa in making data-driven decisions. EasyVisa should leverage these insights to:
1.  **Prioritize Applications**: Focus resources on applications that align with the high-certification profile (e.g., higher education, prior experience, yearly wage, specific continents/regions).
2.  **Implement the Predictive Model**: Integrate the tuned XGBoost model into their workflow to provide a probability of certification, guiding officers to applications that require more detailed scrutiny.
3.  **Enhance Data Collection**: Ensure robust and accurate data collection, especially for highly influential features such as education, region of employment, and wage unit, to maintain and improve model accuracy.
4.  **Investigate Outliers**: Further explore data points flagged as outliers (e.g., very low or high prevailing wages, extremely old company establishment years) to understand their context and ensure they are handled appropriately.
5.  **Continuous Monitoring**: Regularly monitor the model's performance on new data and retrain it periodically to adapt to evolving immigration patterns and regulations, ensuring its long-term accuracy and relevance.