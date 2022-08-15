'''authors: Ester Moiseyev 318692464, Yarden Dali 207220013'''
import numpy as np
import pandas as df

class model():
    '''out implementation to decision tree classifier'''

    def __init__(self, df, cls_col):
        self.df = df
        self.cls_col = cls_col

    def calc_entropy(self, feature_value_data, classification_lst):
        ''' method to calc entropy of data'''
        class_count = feature_value_data.shape[0]
        entropy = 0
        for c in classification_lst:
            label_class_count = feature_value_data[feature_value_data[self.cls_col] == c].shape[
                0]  # row count of class c
            entropy_class = 0
            if label_class_count != 0:
                prob_class = label_class_count / class_count  # probability of the class
                entropy_class = - prob_class * np.log2(prob_class)  # entropy
            entropy += entropy_class
        return entropy

    def info_gain(self, attributes_name, class_list):
        '''method to calc info_gain of a column'''
        attrs_value_list = self.df[attributes_name].unique()
        total_row = self.df.shape[0]
        attributes_info = 0.0
        for attr_value in attrs_value_list:
            attrs_value_data = self.df[self.df[attributes_name] == attr_value]
            attrs_value_count = attrs_value_data.shape[0]
            attrs_value_entropy = self.calc_entropy(attrs_value_data, class_list)
            attrs_value_prob = attrs_value_count / total_row
            attributes_info += attrs_value_prob * attrs_value_entropy
        return self.calc_entropy(self.df, class_list) - attributes_info

    def find_most_informative_attributes(self, class_list):
        '''finding the column with the max value in the calculation of the info gain'''
        attributes_list = self.df.columns.drop(self.cls_col)
        max_info_gain = -1
        max_info_attribute = None

        for attr in attributes_list: #list of columns name without the classification column
            attrs_info_gain = self.info_gain(attr, class_list)#running the info gain on each column
            if max_info_gain < attrs_info_gain:
                max_info_gain = attrs_info_gain
                max_info_attribute = attr
        return max_info_attribute

    def generate_sub_tree(self, attribute_name, class_list):
        '''method to make a sub-tree.
         attribute_name- the name of the column with the biggest info gain
         class_list - the classification column'''
        attr_value_count_dict = self.df[attribute_name].value_counts(sort=False) #counts the  amount of the unique
        # values in the column that has the biggest info gain and saves it in attr_value_count_dict
        tree = {}  # dictunary that will represent our tree

        for attr_value, count in attr_value_count_dict.iteritems(): #for every unique value in this spesific column, we iterate over them and
            attr_value_data = self.df[self.df[attribute_name] == attr_value]
            assigned_to_node = False
            for c in class_list: #run over the classification column
                class_count = attr_value_data[attr_value_data[self.cls_col] == c].shape[0]

                if class_count == count:
                    tree[attr_value] = c
                    self.df = self.df[self.df[attribute_name] != attr_value]
                    assigned_to_node = True
            if not assigned_to_node:
                tree[attr_value] = "?"
        return tree, self.df

    def make_tree(self, root, prev_attr_value, df, classification_lst):
        '''recursive method to make a decision tree'''
        if df.shape[0] != 0: #checks if the data is not empty
            max_info_attribute = self.find_most_informative_attributes(classification_lst)
            tree, df = self.generate_sub_tree(max_info_attribute, classification_lst)
            next_root = None

            if prev_attr_value != None:
                root[prev_attr_value] = dict()
                root[prev_attr_value][max_info_attribute] = tree
                next_root = root[prev_attr_value][max_info_attribute]
            else:  # add to root of the tree
                root[max_info_attribute] = tree
                next_root = root[max_info_attribute]
            print("44444444444444444444444444444444")
            for node, branch in list(next_root.items()):  # iterating the tree node
                if branch == "?":  # if it is expandable
                    attr_value_data = df[df[max_info_attribute] == node]  # using the updated dataset
                    self.make_tree(next_root, node, attr_value_data, classification_lst)

    def id3(self, df_m, label):
        '''id3 algorythm for decision tree
        return: decision tree model (dictionary)'''
        self.df = df_m.copy()
        tree = {}
        classification_lst = self.df[label].unique()
        self.make_tree(tree, None, df_m, classification_lst)
        return tree

    def predict_(self,tree,statement):  #statement =dictionary
        ''' recursive method to predict class value
        return: class val'''
        if type(tree) is dict: # if not a leaf
            root_node = next(iter(tree))  # step
            value = statement[root_node]
            if value in tree[root_node]:
                return self.predict_(tree[root_node][value], statement)
            else:
                return self.df[self.cls_col].mode()
        else:
            print(type(tree))
            return tree



    def predict(self,X_test,y_test):
        ''' method to predict the test file'''
        statement = {}
        lst = []
        tree = self.id3(self.df, self.cls_col)
        new_X_test = X_test.reset_index()
        rows_count = new_X_test.shape[0]
        for row in range(rows_count):
            for col in new_X_test.columns: #making statements
                if col != "index":
                    statement[col] = new_X_test[col][row]
            lst.append(self.predict_(tree, statement)) #predict for each statement
        pre_lst =[x for x in lst if type(x) is np.int32]
        return pre_lst