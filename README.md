# BreastCancerWisconsin-SVM-Matlab
This repository features an SVM classifier implemented using MATLAB for the famous Breast Cancer Wisconsin Dataset. Accuracy achieved 98.5%

I made this for students new to machine learning and the MATLAB software as simple introduction. The code has been writen using the Live Script functionality in MATLAB and should work with any recent MATLAB versions

The code prepares and processes the dataset from a csv file, it performs feature analysis and ranking of features then optimizes the SVM model using a bayesian optimizer. The model is evaluated using K-fold cross-validation, theres is no hold out validation performed. A confusion matrix is also generated to produce the recall,precision and F1 score of the model.

The code is properly structured, well commented and symbolic variables used allowing beginners too have an easy time understanding it.

Please also do cite my work if you like it :-)

# Dataset Brief
This dataset was created to automate breast cancer detection from digitized images of a fine needle aspirate (FNA). An FNA is a biospys procedure conducted to extract cells within a needle from a detected mass within a breast. The FNA cells are then closely investigated in order to diagnose if the mass is benign or cancerous. The data consists of data extracted from a digitized FNA image taken from 569 women whom each was detected with a mass within their breasts. The data has a total of 30 features which describe the characteristics of the cell nuclei present in the image. Each data sample is then placed as either malignant or benign.

Dataset Citation and Source
Dr. William H. Wolberg, W. Nick Street and Olvi L. Mangasarian (1995). UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)]. Irvine, CA: University of California, School of Information and Computer Science. 
