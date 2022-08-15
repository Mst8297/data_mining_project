'''authors: Ester Moiseyev 318692464, Yarden Dali 207220013'''
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder
from math import log2
from sklearn import metrics
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import entropy_based_binning as ebb
import sys
sys.setrecursionlimit(10000)


def valid_info(filename, flag1,class_col):
    '''This function checks if the user enterd the path of the file name and if he chose number of experiment
    also it checks if the user entered the name of the classification column'''
    if filename == "" or flag1 == 0 or class_col == "":
        str = ""
        if filename == "":
            str += "enter the path of the file\n"
        if flag1 == 0: #if non of the radio bottuns where clicked
            str += "choose the number of the experiment\n"
        if class_col == "":
            str += "enter the name of the classification column\n"
        messagebox.showerror("Error", "Error message for user\n" + str)
    else: #the user chose and enterd all the information needed about the data
        return True

def data_split(data,cls_col,value):
    '''This function split the data into train test '''
    X = data.copy()
    y = X.pop(cls_col)
    print(data.info())
    if value == 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.2)
    if value == 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)
    if value == 3:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.4)
    return  X_train, X_test, y_train, y_test

def fill_empty_cls(data,cls_col):
    '''This function fill the empty cells according the classification column'''
    data = data.sort_values(by=cls_col)
    d = dict(data.groupby(by=[cls_col]).size()) #returns {'e': 4208, 'p': 3916}
    for val in d.keys(): #we run over the unique values in the classification column
        for col in data.columns:
            if col == cls_col:
                continue
            else:

                if data[col].dtype == np.int64 or data[col].dtype == np.int32 or data[col].dtype == np.float64:
                    # if the values in the columns are nominal replace the nan values with the mean of this column
                    x = data.groupby(by=[cls_col]).get_group(val)[col].mean()
                    data[col]= data[col].fillna(str(x[0]))
                else:#replace the nan values with the mode of the column depending on the classification column
                    x = data.groupby(by=[cls_col]).get_group(val)[col].mode()
                    data[col]= data[col].fillna(str(x[0]))
    print(data.info())
    #data.to_csv("filled_data.csv")

def fill_empty( class_col,data):
    '''This function help us to fill the empty cells according to the values in each column'''
    print(data.info())
    r = data.shape[0]  # number of rows
    for col in data.columns:
        if col is class_col:
            continue
        if data[col].dtype == np.int64 or data[col].dtype == np.int32 or data[col].dtype == np.float64:
            # if the values in the columns are nominal replace the nan values with the mean of this column
            x = data[col].mean()
            data[col].fillna(x)
        else:#replace the nan values with the mode of this column
            x = str(data[col].mode())
            data[col].fillna(x)
    print(data.info())
    print(data.head())



def norm(data):
    '''This function responsible on the data normalization proccess, we normalize the data with the min-max-Scaler'''
    x=data.values
    y=list(data.columns)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled=min_max_scaler.fit_transform(x)
    data=pd.DataFrame(x_scaled)
    data.columns=y
    return data

def bins_valid_info(value,binsNum):
    '''This function gets the value of the radio button that was clicked end the number of bins that was entered'''
    str = ""
    if value == 0 or binsNum == "":
        if value == 0: #non of the radio button were clicked
            str += " \nChoose type of discritization\n"
        if binsNum == "": #the number of bins for the discretization wasn't entered
            str += "\nEnter the number of bins\n"
        messagebox.showerror("Error", "Error message for user\n" + str)
        return False
    if int(binsNum)<=0:
        str+= "\n choose a number that is bigger then 0\n"
        messagebox.showerror("Error", "Error message for user\n" + str)
    else:
        return True #all the needed information was entered


def Equal_d(data,k):
    '''This function apply the equal-depth discretization, it gets as an argument the data and the number of bins
    we want to devide our data into'''
    for col in data.columns:
        data[col] = pd.qcut(data[col], q=k,duplicates="drop").astype(str)
    data = encode_data(data)
    return data


def Equal_w(data, k):
    '''This function apply the equal-width discretization, it gets as an argument the data and the number of bins
    we want to devide our data into'''
    for col in data.columns:
        min_value = data[col].min()
        max_value = data[col].max()
        bins = np.linspace(min_value, max_value+1, k)
        labels = data[col].unique().tolist()
        if len(labels)+1==k:
            data[col]=pd.cut(data[col],bins=bins, labels= labels, include_lowest=True).astype(str)
        else:
            data[col]=pd.cut(data[col],bins=bins, include_lowest=True).astype(str)
    data = encode_data(data)
    return data

def encode_data(data):
    '''this function responsible for the encoding of the data so that it will be easier to work with it'''
    le = preprocessing.LabelEncoder()
    for col in data.columns:
        if data[col].dtype == object:
            data[col]=le.fit_transform(data[col])

def get_accuracy(data ):
    '''this function help us to calculate the accuracy of the model we were running on our data'''
    y_pred = []
    count = 0
    for r in x_test.shape[0]:
        t_pred.append(predict(x_test.iloc[r]))
    for i in range(x_text.shape[0]):
        if y_pred[i] == y_text[i]:
            count = count + 1
    return count / x_test.shape[0]


def knn_m(n,y_train, X_train,X_test,y_test):
    '''This function runs the KNN model. It gets the number of neighbors and the y_train, X_train,X_test,y_test'''
    try:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        report(y_test,y_pred)
        # save the model to disk
        filename = 'finalized_model_knn-60-40.sav'
        joblib.dump(knn, filename)
        # load the model from disk
        loaded_model = joblib.load(filename)
    except ValueError as ve:
        messagebox.showerror("Error!", ve)
        #messagebox.showerror("Error!","the neighbors number should be less then " + str(len(X_train))+ "and more then 0")



def kMean_clus(clusters,X_train, y_train, X_test, y_test):
    '''this function apply the Kmeans clustering on our data, it gets
    the number of clusters we want to divide to and the y_train, X_train,X_test,y_test'''
    try:
        model = KMeans(clusters)
        model.fit(X_train, y_train)
        identified_clusters = model.fit(X_train)
        y_pred = identified_clusters.predict(X_test)
        X_train['Clusters'] = identified_clusters
        filename = 'finalized_model_kmeans.sav'
        joblib.dump(model, filename)
        report(y_test,y_pred)
        # load the model from disk
        loaded_model = joblib.load(filename)
        return X_train

    except ValueError as ve:
        messagebox.showerror("Error!", ve)
    return data


def entropy_based_binning(data,number_of_bins):
    '''this function calculate the entropy based binning'''
    return ebb.bin_array(data,nbins=number_of_bins,axis = 1)

def combine_2_df(data1,data2,cls_col):
    '''this function gets 2 dataframes and combine them together'''
    data = data1.tolist()
    df = pd.DataFrame(data, columns=[cls_col])
    df = pd.concat([df, data2], axis=1, join='inner')
    new_df=df.reset_index()
    #new_df.to_csv("newdf.csv",index = False)
    return new_df


def similarity(lst,y_test):
    '''this function gets the list with the predicted classification column and the real classification column
    it compare both of them and return the similarity precent between them'''
    count=0
    y_lst = list(y_test)
    i=0
    for r in y_test.values:#runs over the values in the classification column
        if lst[i]==r:
           count +=1
        i+=1
    return (count/len(lst))*100


def report(y_test, y_pred):
    '''this function givven the classification column of the test data and the classification column that was predicted,
     calculate the confusion matrix ,the accuracy score and shows us the classification report'''
    con_matrix = confusion_matrix(y_test, y_pred)
    result = accuracy_score(y_test, y_pred) * 100
    cls_report = classification_report(y_test, y_pred)
    print("Confusion Matrix: \n", con_matrix)
    print("Accuracy : ", result)
    print("Classification report:", cls_report)