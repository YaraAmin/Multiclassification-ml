# Multiclassification-ml
Documentation
RESULTS
----------- SVM Clasiifier --------------
Total execution time = 0.0625009536743164
Ang. accuracy = 0.879
F-score​ = 0.82
----------- Naive Bayes Clasiifier ---------
Total execution time = 0.01563429832458496
Ang. accuracy = 0.875
​F-score​=​0.81
I think its logical results can be improved by using more effectient feature selection and bigger
number of n-compunent at classifier SVM.Documentation


First, i uploaded the data as json file, converted it to data frame to can use it, faced a problem that
the values in the columns in lists and i was unable to use it at any step of my model, this honsetly
took too much time, finally managed to solve it by using df.apply()


Second i visualized the data and calc sum of culomns that have nulls and hen drop it, i can cala
average or mean and replace the null but its number was bigger than the half number of the data
so that would ruin our data


Drop the label culomn and save it to y


Get the rest values of the rest data frame and assign it to X


Split the data to train and test set


Fit it to LDA feature selection


Make a pipline of PCA and Classifiers -SVM, Naive bayesFit the output to the pipline


