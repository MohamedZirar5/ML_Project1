# CLASS IMBALANCE

**3|4 ways how to deal with it:**
- random undersampling (pretty bad idea hihihihi)
- random oversampling (pretty simple and stupid idea probably)
- oversampling w SMOTE (artificial creation of new samples for imbalanced class... the best idea so far)
- Upweight the downsampled class
- oversampling w ADASYN
- Undersampling with Tomek Link
- Oversampling with SMOTE, then undersample with TOMEK Link (SMOTE-Tomek)



# APPLY EVERYTHING AND THAN TEST IT WITH ROC CURVE AND CHOOSE THE BEST ONE?


# for thsi scenario its quite normal that its imbalanced bcs heart disease is rare... so should i judge only f





Specifically, logistic regression combined with the synthetic minority oversampling technique (SMOTE) produces the best detection results, with a significant 45.3% increase in recall (recall = 75.7%) compared with pure logistic regression (recall = 52.1%).
(source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5987310/)