'''authors: Ester Moiseyev 318692464, Yarden Dali 207220013'''
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import myProject as mp
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import naive_bayes_model
import decision_tree_model
from sklearn import model_selection
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



flag1=False
filename=""
window = Tk()
window.title("My project")
window.geometry('500x450')
X_train= None
X_test = None
y_train =None
y_test = None
cls_col = ""
cls_flag = False

def browseFiles():
    '''This funcion open the browser file window so that the user will be able to choose the data file,returns the filename '''
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",title = "Select a File",filetypes = (("Text files","*.txt*"),("all files","*.*")))
    label_file_explorer.configure(text="File Opened: " + filename)
    return filename
label_file_explorer = Label(window,text = "Choose the file path",width = 100, height = 4,fg = "blue")
button_explore = Button(window,text="Browse Files",command=browseFiles)
label_file_explorer.pack()
button_explore.pack()
exp_lbl = Label(window,text="choose the train test devision you would like to check:", font =('Arial',8, 'bold'))
exp_lbl.pack()
r=IntVar()


def clicked1(value):
    '''if a specific radio button is clicked it will turn the flag value into 1 so that it wont raise an error messagebox'''
    global flag1
    flag1=1

'''the radio buttons that says which split to do'''
rb1 = Radiobutton(window,text="Train- 80% \n Test- 20%", variable=r,value=1, command =lambda:clicked1(r.get()))
rb2 = Radiobutton(window,text="Train- 70% \n Test- 30%", variable=r,value=2, command =lambda:clicked1(r.get()))
rb3 = Radiobutton(window,text="Train- 60% \n Test- 40%", variable=r,value=3, command =lambda:clicked1(r.get()))
rb1.pack()
rb2.pack()
rb3.pack()
classification_lbl=Label(window, text="Enter the name of the classification column:", font =('Arial',8, 'bold'),fg='red')
classification_lbl.pack()
classification_tb = Text(window, width=20, height=1)
classification_tb.pack()
enter_bt= Button(window,text="Continue",command=lambda: openNewWindow(r.get()) )
enter_bt.pack()


def openNewWindow(value):
    ''' Here we check if the classification column name that was given is valid
     and here we turn the csv file into a dataframe'''
    global cls_flag
    global flag1
    global filename
    global cls_col
    global data
    if mp.valid_info(filename,flag1,classification_tb.get("1.0", "end-1c"))==True:
        data = pd.read_csv(filename)
        cls_col = classification_tb.get("1.0", "end-1c")
        for col in data.columns:
            if col == cls_col:
                cls_flag = True #if the column name given by the user exist in the dataframe
        if cls_flag == False:
            messagebox.showerror("Error", "Column's not found! \n try again!")
            return
        ##deleting rows with no value in the classification column
        data = data[data[classification_tb.get("1.0", "end-1c")].notna()]  # has no value
        data.dropna(how='all')  # has NAN
        data = data.replace("?", np.NaN) #replace all the "?" into nan values
        #data.to_csv("data.csv")
        newWindow = Toplevel(window)
        newWindow.title("Preproccecing")
        newWindow.geometry('500x450')

        t1 = IntVar()
        t2 = IntVar()
        t3 = IntVar()
        #the choises of how to preproces the data are showen as a radio button
        missing_value_lbl = Label(newWindow, text="choose how you'de like to fill the cells with no value:")
        fill_rb1 = Radiobutton(newWindow, text="fill them according the classification column", variable=t1, value=1)
        fill_rb2 = Radiobutton(newWindow, text="not to fill them according the classification column", variable=t1, value=2)
        approve1_bt = Button(newWindow, text="Approve", command=lambda: buttonClick2(t1.get()))
        normalization_lbl = Label(newWindow, text="Choose if a normalization is needed")
        fill_rb3 = Radiobutton(newWindow, text="Yes", variable=t2, value=3)
        fill_rb4 = Radiobutton(newWindow, text="No", variable=t2, value=4)
        approve2_bt = Button(newWindow, text="Approve", state=DISABLED, command=lambda: buttonClick2(t2.get()))
        discrit_lbl = Label(newWindow, text="Choose if a discritization is needed")
        fill_rb5 = Radiobutton(newWindow, text="Yes", variable=t3, value=5)
        fill_rb6 = Radiobutton(newWindow, text="No", variable=t3, value=6)
        approve3_bt = Button(newWindow, text="Approve", state=DISABLED, command=lambda: buttonClick2(t3.get()))

        missing_value_lbl.pack()
        fill_rb1.pack()
        fill_rb2.pack()
        approve1_bt.pack()
        normalization_lbl.pack()
        fill_rb3.pack()
        fill_rb4.pack()
        approve2_bt.pack()
        discrit_lbl.pack()
        fill_rb5.pack()
        fill_rb6.pack()
        approve3_bt.pack()


        def buttonClick2(value):
            '''this is the function that strarts run when we click the approve buttons in our program'''
            global X_train
            global X_test
            global y_train
            global y_test
            if value == 1:#fill the missing values according the classification column
                mp.fill_empty_cls(data,cls_col)
                mp.encode_data(data)
                X_train, X_test, y_train, y_test=mp.data_split(data, cls_col,r.get())
                #data.to_csv('clean_data.csv', index=False)
                approve2_bt['state'] = NORMAL
            if value == 2: #fill the missing values according the values in each column
                mp.fill_empty(cls_col,data)
                mp.encode_data(data)
                X_train, X_test, y_train, y_test = mp.data_split(data, cls_col, r.get())
                #data.to_csv('clean_data.csv', index=False)
                #y_train.to_csv("y.csv", index =False)
                print(data)
                print(data.info())
                approve2_bt['state']=NORMAL

            if value == 3:#activate the normalization on the data
                X_train = mp.norm(X_train)
                X_test = mp.norm(X_test)
                #X_train.to_csv("train.csv", index=False)
                approve3_bt['state'] = NORMAL

            if value == 4: #in case the user doesnt want to normalize the data
                approve3_bt['state'] = NORMAL
            if value==5:# if the user want to run a descritization on the data
                openNewWindow_2(X_train,X_test,y_train, y_test)#opens a new window to choose which type of descritization to run
            if value == 6: #in case no discritization is needed
                openNewWindow_3()#open the page with the model chosing

    def openNewWindow_2(X_train,X_test,y_train, y_test):
        '''difine the window where we choose the type of descritization'''
        newWindow2 = Toplevel(newWindow)
        newWindow2.title("Descritization")
        newWindow2.geometry('500x450')
        d1=IntVar()
        title_lbl = Label(newWindow2, text = "choose type of discritization:")
        dis_rb1=Radiobutton(newWindow2,text= "Equal frequency discretization",variable= d1, value=1)
        dis_rb2=Radiobutton(newWindow2,text= "Equal width discretization",variable= d1, value=2)
        dis_rb3=Radiobutton(newWindow2,text= "Descritization based on entropy",variable= d1, value=3)
        bins_lbl = Label(newWindow2, text="Enter the number of bins you would like to have:",font=('Arial', 8, 'bold'), fg='red')
        bins_tb = Text(newWindow2, width=10, height=1)
        dis_bt = Button(newWindow2,text="next", command= lambda: clicked_button_d(d1.get(),bins_tb.get("1.0", "end-1c")))

        title_lbl.pack()
        dis_rb1.pack()
        dis_rb2.pack()
        dis_rb3.pack()
        bins_lbl.pack()
        bins_tb.pack()
        dis_bt.pack()


        def clicked_button_d(value,binsNum):
            '''this function activate the function needed accordig the users choices'''
            global X_train
            global cls_col
            global X_test
            global data
            binsNum1 = int(binsNum)
            if mp.bins_valid_info(value,binsNum)==True: #if the number of bins is okay, we strat with the functions
                if value==1: #activate the equal depth descritization
                    new_train = mp.Equal_d(X_train,binsNum1)
                    new_test = mp.Equal_d(X_test,binsNum1)
                if value==2: #activate the equal width descritization
                    new_train = mp.Equal_w(X_train, binsNum1)
                    new_test = mp.Equal_w(X_test, binsNum1)
                if value ==3: #activate the entropy based binning
                    data = mp.entropy_based_binning(data, binsNum1)
                openNewWindow_3()# after the function was activated we open the next window of the models
    def openNewWindow_3():
        newWindow3 = Toplevel(newWindow)
        newWindow3.title("Model choosing")
        newWindow3.geometry('500x450')
        m1=IntVar()
        c=IntVar()
        global cls_col

        def pick_model(value):
            '''this function is activate the right function according the user's choices'''
            if value == 1 or value == 2: #decision tree, our implementation
                dt_my_rb1['state'] = NORMAL
                dt_lib_rb2['state'] = NORMAL
                q_tb['state'] = DISABLED
            if value == 3 or value == 4:
                q_tb['state'] = NORMAL
                dt_my_rb1['state'] = DISABLED
                dt_lib_rb2['state'] = DISABLED

        model_choosing = Label(newWindow3, text="Choose the model you want to run on your data(1-4):\n")
        model_rb1 = Radiobutton(newWindow3, text="1-Decision tree model", variable=m1, value=1, command=lambda: pick_model(m1.get()))
        model_rb2 = Radiobutton(newWindow3, text="2-Naive bayes", variable=m1, value=2, command=lambda: pick_model(m1.get()))
        imp_choosing = Label(newWindow3, text="\nChoose the the type of implementation (only for models 1,2):")
        dt_my_rb1 = Radiobutton(newWindow3, text="our implementation", variable=c, value=1, state=DISABLED)
        dt_lib_rb2 = Radiobutton(newWindow3, text="library implementation", variable=c, value=2, state=DISABLED)
        model_rb3 = Radiobutton(newWindow3, text="3-Knn model", variable=m1, value=3,command=lambda: pick_model(m1.get()))
        model_rb4 = Radiobutton(newWindow3, text="4-K-means model", variable=m1, value=4,command=lambda: pick_model(m1.get()))
        cluster_lbl = Label(newWindow3, text="Enter the number of clusters/neighbors (only for models 3,4)\n")
        q_tb = Text(newWindow3, width=10, height=1, state = DISABLED)

        next_bt = Button(newWindow3, text="next", command=lambda: button_clicked_m(m1.get(), c.get(),q_tb.get("1.0", "end-1c"),newWindow3))

        model_choosing.pack()
        model_rb1.pack()
        model_rb2.pack()
        imp_choosing.pack()
        dt_my_rb1.pack()
        dt_lib_rb2.pack()
        model_rb3.pack()
        model_rb4.pack()
        cluster_lbl.pack()
        q_tb.pack()
        next_bt.pack()


    def button_clicked_m(value_m,value_c,cn,newWindow3):
        '''definition of a button- about the choosing of the models, thos button apply the changes on the dataframe according the choises the user made with the radio buttons
        value_m-argument that responsible on the type of model, value_c -responsible on the type of implementation(ours or library), cn- holds the number of clusters needed for the kmeans, and the number of neighbors for the knn '''
        global y_train
        global cls_col
        global X_test
        global y_test
        if value_m== 1 and value_c==1: #decision tree model-our implementation
            model1=decision_tree_model.model(data, cls_col)
            y_pred = model1.predict(X_test, y_test)
            mp.report(y_test, y_pred) #model results
        if value_m == 1 and value_c == 2: #decision tree model-library implementation
            clf_gini = DecisionTreeClassifier(criterion="gini",random_state=100, max_depth=3, min_samples_leaf=5,max_features=22)
            # Performing training
            try:
                clf_gini.fit(X_train, y_train)
            except ValueError as ve:
                messagebox.showerror("Error", ve)
            y_pred = clf_gini.predict(X_test)
            mp.report(y_test,y_pred)
            filename = 'finalized_model_dt_70_30.sav'
            joblib.dump(clf_gini, filename)
            loaded_model = joblib.load(filename)
            result = loaded_model.score(X_test, y_test)
            print(result)
        if value_m == 2 and value_c == 1: #NB-our implementation
            #mp.get_accuracy(data)
            df = mp.combine_2_df(y_train, X_train,cls_col)
            #df.to_csv("new_df.csv", index=False)
            model1 = naive_bayes_model.model(df, cls_col)
            # save the model to disk
            filename = 'finalized_model_nb1_60_40.sav'
            joblib.dump(model1, filename)
            # load the model from disk
            loaded_model = joblib.load(filename)
            result = loaded_model.predict(X_test, y_test)
            mp.report(y_test, result)
            print(result)
        if value_m == 2 and value_c == 2:
            gnb = GaussianNB()
            try:
                gnb.fit(X_train, y_train)
            except ValueError as ve:
                messagebox.showerror("Error", ve)
            y_pred = gnb.predict(X_test)
            # save the model to disk
            filename = 'finalized_model_nb2_70_30.sav'
            joblib.dump(gnb, filename)
            # load the model from disk
            loaded_model = joblib.load(filename)
            mp.report(y_test,y_pred)

        if value_m == 3:
            try:
                mp.knn_m(int(cn), y_train, X_train,X_test,y_test)
            except ValueError as ve:
                messagebox.showerror("Error", ve)

        if value_m == 4:
            try:
                d=mp.kMean_clus(int(cn),X_train, y_train, X_test, y_test)
                #d.to_csv("data_with_clusters.csv")
            except ValueError as ve:
                messagebox.showerror("Error", ve)



window.mainloop()