#!/usr/bin/env python
# coding: utf-8

# In[17]:


def load_data():
    df = pd.read_json(test_data.json',lines=True)
    df.shape
    df.describe()
    df.head()
    #visualize and select features
    df.isnull().sum() #check if there are any missing values, found (c3,c5,c6,c9) have nulls
    #drop columns contains nulls
    df.drop(columns="c3",inplace=True)
    df.drop(columns="c5",inplace=True)
    df.drop(columns="c6",inplace=True)
    df.drop(columns="c9",inplace=True)
    df.head()
    #get labels
    y = df['l'].values
#     print(y,y.shape)
    df.drop(columns="l",inplace=True)
    def func(x):
        return x[0]
    df['c0'] = df['c0'].apply(func)
    df['c1'] = df['c1'].apply(func)
    df['c2'] = df['c2'].apply(func)
    df['c4'] = df['c4'].apply(func)
    df['c7'] = df['c7'].apply(func)
    df['c8'] = df['c8'].apply(func)
    df.head()
    #Selecting features -LDA-
    x=df.values
#     print(x)
    # pd.DataFrame(np.concatenate(list_arrays))
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)  
    X_test = sc.transform(X_test)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    lda = LDA(n_components=1)  
    X_train = lda.fit_transform(X_train, y_train)  
    X_test = lda.transform(X_test)  


#     #split data into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                      test_size=0.20,
#                      stratify=y,
#                      random_state=1)
    return X_train, X_test, y_train, y_test
    
    
#--------------------------------------------------------------------
# #Selecting features -genetic-
# def GA(X_train, X_test, y_train, y_test):
#     from deap import creator, base, tools, algorithms
#     from scoop import futures
#     import random
#     import numpy
#     from scipy import interpolate
#     allFeatures = df[:]
#     allFeatures.shape
#     allClasses=y
#     X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(allFeatures, allClasses, test_size=0.20, random_state=42)
#     X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20, random_state=42)
#     # Feature subset fitness function
#     def getFitness(individual, X_train, X_test, y_train, y_test):

#     # Parse our feature columns that we don't use
#     # Apply one hot encoding to the features
#     cols = [index for index in range(len(individual)) if individual[index] == 0]
#     X_trainParsed = X_train.drop(X_train.columns[cols], axis=1)
#     X_trainOhFeatures = pd.get_dummies(X_trainParsed)
#     X_testParsed = X_test.drop(X_test.columns[cols], axis=1)
#     X_testOhFeatures = pd.get_dummies(X_testParsed)

#     # Remove any columns that aren't in both the training and test sets
#     sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
#     removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
#     removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
#     X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
#     X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)

#     # Apply logistic regression on the data, and calculate accuracy
#     clf = LogisticRegression()
#     clf.fit(X_trainOhFeatures, y_train)
#     predictions = clf.predict(X_testOhFeatures)
#     accuracy = accuracy_score(y_test, predictions)

#     # Return calculated accuracy as fitness
#     return (accuracy,)

#     #========DEAP GLOBAL VARIABLES (viewable by SCOOP)========

#     # Create Individual
#     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     creator.create("Individual", list, fitness=creator.FitnessMax)

#     # Create Toolbox
#     toolbox = base.Toolbox()
#     toolbox.register("attr_bool", random.randint, 0, 1)
#     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(df.columns) - 1)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#     # Continue filling toolbox...
#     toolbox.register("evaluate", getFitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#     toolbox.register("mate", tools.cxOnePoint)
#     toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
#     toolbox.register("select", tools.selTournament, tournsize=3)

#     def getHof():

#     # Initialize variables to use eaSimple
#         numPop = 100
#         numGen = 10
#         pop = toolbox.population(n=numPop)
#         hof = tools.HallOfFame(numPop * numGen)
#         stats = tools.Statistics(lambda ind: ind.fitness.values)
#         stats.register("avg", numpy.mean)
#         stats.register("std", numpy.std)
#         stats.register("min", numpy.min)
#         stats.register("max", numpy.max)

#         # Launch genetic algorithm
#         pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof, verbose=True)

#         # Return the hall of fame
#         return hof

#     def getMetrics(hof):

#     # Get list of percentiles in the hall of fame
#         percentileList = [i / (len(hof) - 1) for i in range(len(hof))]

#         # Gather fitness data from each percentile
#         testAccuracyList = []
#         validationAccuracyList = []
#         individualList = []
#         for individual in hof:
#             testAccuracy = individual.fitness.values
#             validationAccuracy = getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
#             testAccuracyList.append(testAccuracy[0])
#             validationAccuracyList.append(validationAccuracy[0])
#             individualList.append(individual)
#         testAccuracyList.reverse()
#         validationAccuracyList.reverse()
#         return testAccuracyList, validationAccuracyList, individualList, percentileList





#     individual = [1 for i in range(len(allFeatures.columns))]
#     testAccuracy = getFitness(individual, X_train, X_test, y_train, y_test)
#     validationAccuracy = getFitness(individual, X_trainAndTest, X_validation, y_trainAndTest, y_validation)
#     print('\nTest accuracy with all features: \t' + str(testAccuracy[0]))
#     print('Validation accuracy with all features: \t' + str(validationAccuracy[0]) + '\n')

#     '''
#     Now, we will apply a genetic algorithm to choose a subset of features that gives a better accuracy than the baseline.
#     '''
#     hof = getHof()
#     testAccuracyList, validationAccuracyList, individualList, percentileList = getMetrics(hof)

#     # Get a list of subsets that performed best on validation data
#     maxValAccSubsetIndicies = [index for index in range(len(validationAccuracyList)) if validationAccuracyList[index] == max(validationAccuracyList)]
#     maxValIndividuals = [individualList[index] for index in maxValAccSubsetIndicies]
#     maxValSubsets = [[list(allFeatures)[index] for index in range(len(individual)) if individual[index] == 1] for individual in maxValIndividuals]

#     print('\n---Optimal Feature Subset(s)---\n')
#     for index in range(len(maxValAccSubsetIndicies)):
#         print('Percentile: \t\t\t' + str(percentileList[maxValAccSubsetIndicies[index]]))
#         print('Validation Accuracy: \t\t' + str(validationAccuracyList[maxValAccSubsetIndicies[index]]))
#         print('Individual: \t' + str(maxValIndividuals[index]))
#         print('Number Features In Subset: \t' + str(len(maxValSubsets[index])))
#         print('Feature Subset: ' + str(maxValSubsets[index]))

#     '''
#     Now, we plot the test and validation classification accuracy to see how these numbers change as we move from our worst feature subsets to the 
#     best feature subsets found by the genetic algorithm.
#     '''
#     # Calculate best fit line for validation classification accuracy (non-linear)
#     tck = interpolate.splrep(percentileList, validationAccuracyList, s=5.0)
#     ynew = interpolate.splev(percentileList, tck)
    






#--------------------------------------------------------------
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def svm_classifier(X_train, y_train,X_test):
    pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=1),
                        SVC())

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    return y_pred

def naive_bayes_classifier(X_train, y_train,X_test) :
    pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=1),
                        GaussianNB())

    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test)
    
    return y_pred


def performance_calc(y_test, y_pred):
    print('Ang. accuracy = ', accuracy_score(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    
    
def ROC_SVM(X_train, y_train,X_test,y_test,y):

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(y, n_folds=6)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state= np.random.RandomState(0))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    

def ROC_NV(X_train, y_train,X_test,y_test,y):
    
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(y, n_folds=6)
    classifier = GaussianNB()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
#------------------------------------------
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
np.random.seed(123) # makes the random numbers predictable 
#from classification import svm_classifier, naive_bayes_classifier, logistic_regression_classifier, performance_calc
#from load_data import load_data
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt


Data_train , Data_test , label_train , label_test = load_data()

print('----------- SVM Clasiifier --------------')
start = time.time()
svm_label_pred = svm_classifier(Data_train, label_train, Data_test)
end = time.time()
print('Total execution time = ', end - start)
performance_calc(label_test, svm_label_pred)
ROC_SVM(Data_train, label_train,Data_test,label_test,label_train)

print('----------- Naive Bayes Clasiifier ---------')
start = time.time()
nb_label_pred = naive_bayes_classifier(Data_train, label_train, Data_test)
end = time.time()
print('Total execution time = ', end - start)
performance_calc(label_test, nb_label_pred)
ROC_NV(Data_train, label_train,Data_test,label_test,label_train)

