# BreastCancerWisconsin-SVM-Matlab
This repository features an SVM classifier implemented using MATLAB for the famous Breast Cancer Wisconsin Dataset. Accuracy achieved 98.5%

I made this for students new to machine learning and the MATLAB software as simple introduction. The code has been writtwen using the Live Script functionality in MATLAB and should work with any recent MATLAB versions

The code prepares and processes the dataset from a csv file, it performs feature analysis and ranking of features then optimizes the SVM model using a bayesian optimizer. The model is evaluated using K-fold cross-validation, theres is no hold out validation performed. A confusion matrix is also generated to produce the recall,precision and F1 score of the model.

The code is properly structured, well commented and symboic variables allowing beginners too have an easy time understanding it.

Dataset Citation
Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian (1995). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)]. Irvine, CA: University of California, School of Information and Computer Science. 
