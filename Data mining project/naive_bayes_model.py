'''authors: Ester Moiseyev 318692464, Yarden Dali 207220013'''
from typing import NoReturn
import myProject as mp

class model():

    '''out implementation to naive bayes classifier'''

    def __init__(self, df, cls_col):
        self.df = df
        self.cls_col = cls_col

    # prior func
    def prior_(self):
        '''Priority of values of class col'''
        return self.df.groupby(self.cls_col).size().div(len(self.df))


        # likelihoods func
    def likelihood(self):
        '''Likelihood calculation
            return: dictionary of likelihoods'''
        priorVal = self.prior_()
        LH = dict()
        for i in self.df.columns:
            if i != "index":
                LH[i] = self.df.groupby([self.cls_col, i]).size().div(len(self.df)).div(priorVal)
            else:
                continue
        del LH[self.cls_col]
        return LH


    def calc_nb(self, statement, cls, lh):
        '''method to predict class value of given statement
            return: prediction'''
        maxP = 0
        p = 0
        lst = []
        for col in statement.keys(): #savnig list of unique values
            lst.append(col)
        for v in cls:
            sum = 1
            i = 0
            for xv in statement.values():
                try:
                    if lh[lst[i]][v][xv] > 0:
                        sum = sum * lh[lst[i]][v][xv] #getting the probability of given value in the data based on column
                        i += 1
                    else:
                        break
                except KeyError:
                    continue
            sum = sum * self.prior_()[v] #calculating the total condition probability
            if maxP < sum:
                maxP = sum
                p = v

        return p #prediction


    # prediction func
    def predict_(self, statement):
        '''model prediction func
            return: prediction for given statement'''

        cls = self.df[self.cls_col].unique()
        prior_ = self.prior_()
        lh = self.likelihood()
        prediction = self.calc_nb(statement, cls, lh)
        return prediction


    def predict(self,X_test,y_test):
        '''calculating accuracy score'''
        statement = {}
        pre_lst = []
        new_X_test = X_test.reset_index()
        rows_count = new_X_test.shape[0]
        for row in range(rows_count):
            for col in new_X_test.columns: #making a statements from test file
                if col != "index":
                    statement[col] = new_X_test[col][row]
            pre_lst.append(self.predict_(statement)) # saving prediction in a list
            print(pre_lst)
        return pre_lst