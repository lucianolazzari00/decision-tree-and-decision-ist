import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_set = pd.read_csv("restaurant_waiting.csv")
id_node = 0

def uniform_value(data):
    yes_flag = True
    no_flag = True
    for sample in data:
        if sample[10] == "No":
            yes_flag = False
        if sample[10] == "Yes":
            no_flag = False
    return yes_flag or no_flag, yes_flag, no_flag

def plurality_value(data):
    yes = 0
    no = 0
    for sample in data:
        if sample[10] == "Yes":
            yes += 1
        else:
            no += 1
    if yes>no:
        return "Yes"
    else:
        return "No"

class node():
    #this will be the nodes of the DT

    def __init__(self, feature=None, i_gain=None, children_list=None, split_value=None, will_wait=None, plurality_value=None):
        self.split_value = split_value
        self.feature = feature
        self.i_gain = i_gain
        self.children_list = children_list
        self.will_wait = will_wait
        self.plurality_value = plurality_value

class decision_tree():
    #this will be the tree

    def __init__(self, maxDepth=3):
        self.maxDepth = maxDepth
        self.root = None
        self.used_index = []

    def learn_tree(self,data,depth,s_val):
        X = data[:, :-1]
        Y = data[:, -1]

        num_feature = len(X[0])
        num_samples = len(X)
        
        u,y,n = uniform_value(data)
        if(depth <= self.maxDepth and num_samples >= 0 and u==False):
            #split
            b_feature, gain, children_datasets= self.best_feature(data,num_feature,num_samples)
            self.used_index.append(b_feature)
            children_list_temp = []
            for k in children_datasets.keys():
                ar = np.asarray(children_datasets[k])
                children_list_temp.append(self.learn_tree(ar,depth+1,k))
            return node(b_feature,gain,children_list_temp,s_val,plurality_value=plurality_value(data))

        else:
            wait = plurality_value(data)
            return node(split_value= s_val, will_wait=wait)


    def best_feature(self,data,num_feature,num_samples):
        #find the best feature to use for split and split the ds
        max_info_gain= -99999
        b_feat_index = 0
        best_splitted_dataset = {}
        for feature_index in range(num_feature):
            if feature_index in self.used_index:
                continue #dont use the same attribite two times!!
            splitted_ds = self.dataset_split(data,feature_index)
            info_gain = self.compute_information_gain(data,splitted_ds)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                b_feat_index = feature_index
                best_splitted_dataset = splitted_ds
        return b_feat_index, max_info_gain, best_splitted_dataset

    def dataset_split(self,data,feature_index):
        feature_values = data[:,feature_index]
        possible_values = np.unique(feature_values)
        splitted_dataset = {}
        for value in possible_values:
            ds=[]
            for sample in data:
                if sample[feature_index] == value:
                    ds.append(sample)
            splitted_dataset[value] = ds
        return splitted_dataset

    def entropy(self,data):
        p = 0
        n = 0
        for sample in data:
            if sample[10] == "Yes":
                p += 1
            elif sample[10] == "No":
                n += 1
        if p==0 or n==0:
            B = 0
            return B,p,n
        q = p/(p+n)
        log_q = np.log2(q)
        log_1_q = np.log2(1-q)
        B = - (q*log_q + (1-q)*log_1_q)
        return B,p,n

    def compute_information_gain(self,original_datset,splitted_dataset):
        #i-gain = entropy_father-remainder = B(p/p+k)-sum( (pk+nk)/(p+n) * B(pk/pk+nk) )
        entr_father,p,n = self.entropy(original_datset)
        rem = 0
        for ds in splitted_dataset.values():
            entr_son,pk,nk = self.entropy(np.asarray(ds))
            weight = (pk+nk)/(p+n)
            rem += weight*entr_son
        return entr_father - rem

    def print_tree(self,node,depth=0,fat=0):
        feature_names = ["Alt","Bar","Fri","Hun","Pat","Price","Rain","Res","Type","Est","Wait"]
        global id_node
        id_curr = id_node
        id_node += 1
        if node.children_list == None:
            print("\t"*depth,node.split_value,"||wait = ",node.will_wait)
            return
        print("\t"*depth,node.split_value,"||",feature_names[node.feature],"?")
        for nodes in node.children_list:
            self.print_tree(nodes,depth+1,id_curr)
    
    def show_dtree(self):
        #wrapper for print_tree()
        print("\n> Decision Tree:\n ")
        self.print_tree(self.root)
        print("\n")

    def predict(self,X,node):
        i = node.feature
        if node.children_list == None:
            return node.will_wait
        for child in node.children_list:
            if child.split_value == X[i]:
                return self.predict(X,child)
        return node.plurality_value

    def make_predictions(self,X):
        predictions = []
        for elem in X:
            predictions.append(self.predict(elem,self.root))
        return predictions

    def fit_model(self,X,Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.learn_tree(dataset,0,"")

class list_node:
    def __init__(self,attr=None,res=None,will_wait=None, nn=None):
        self.cond_attribute = attr  #the attribute(s) that we have to test
        self.cond_result = res      #the answer of the test
        self.will_wait = will_wait  #the outcome of the test
        self.next_node = nn
    
class decision_list:
    def __init__(self,list_r=None,max_length=20):
        self.list_root = list_r
        self.max_length = max_length
    
    def learn_list(self,data):
        if len(data) == 0:
            return
        X = data[:, :-1]
        Y = data[:, -1]

        num_feature = len(X[0])
        num_samples = len(X)
        if num_samples == 0: #data is empty
            return list_node(will_wait="No")
        a,r,o= self.find_test_attribute(data,num_feature)
        new_ds = []
        for sample in data:
            if sample[a]!=r:
                new_ds.append(sample)
        if a==-1 and r ==-1 and o =="nomatch":
            return -1
        curr_node = list_node(a,r,o)
        curr_node.next_node = self.learn_list(np.asarray(new_ds))
        return curr_node
    
    def find_test_attribute(self,data,num_feature):
        #find the attribute
        #remove from data the matching examples
        for feat in range(num_feature):
            feature_values = data[:,feat]
            possible_values = np.unique(feature_values)
            for val in possible_values:
                new_ds = self.select_dataset(data,feat,val) #select* from data where feat=val
                uniform,yes_f,no_f = uniform_value(new_ds)
                if uniform:
                    if yes_f:
                        return feat,val,"Yes"
                    elif no_f:
                        return feat,val,"No"
        return -1,-1,"nomatch"

    def select_dataset(self,data,feature_index,value,complementary=False):
        new_ds = []
        for sample in data:
            if complementary==False:
                if sample[feature_index]==value:
                    new_ds.append(sample)
            else:
                if sample[feature_index]!=value:
                    new_ds.append(sample)
        return new_ds

    def make_predictions():
        return 0

    def fit_list(self,X,Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.list_root = self.learn_list(dataset)

    def predict(self,X):
        root=self.list_root
        while True:
            if root.next_node==None:
                return root.will_wait
            if X[root.cond_attribute]==root.cond_result:
                return root.will_wait
            root = root.next_node
        return -1
    
    def make_predictions(self,X):
        Y_pred = []
        for sample in X:
            Y_pred.append(self.predict(sample))
        return Y_pred

    def print_list(self):
        feature_names = ["Alt","Bar","Fri","Hun","Pat","Price","Rain","Res","Type","Est","Wait"]
        print("\n> Decision List:\n")
        root = self.list_root
        while True:
            if(root.next_node==None):
                print("[wait=",root.will_wait,"]")
                break
            print("[if(",feature_names[root.cond_attribute],")is equal to(",root.cond_result,"): wait=",root.will_wait,"]...")
            root = root.next_node
        print("\n")

    
        
tree = decision_tree(4)
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=35)
tree.fit_model(X_train,Y_train)
tree.show_dtree()
Y_pred = tree.make_predictions(X_test)
print("> X test set:\n",X_test,"\n")
print("> Y predicted by model:\n",Y_pred,"\n")
print("> Y test set:\n",Y_test.reshape(1,-1)[0],"\n")
print("Accuracy: ",accuracy_score(Y_test, Y_pred))

print("======================================================")

dlist = decision_list()
dlist.fit_list(X_train,Y_train)
dlist.print_list()
Y_pred2 = dlist.make_predictions(X_test)
print("> X test set:\n",X_test,"\n")
print("> Y predicted by model:\n",Y_pred2,"\n")
print("> Y test set:\n",Y_test.reshape(1,-1)[0],"\n")
print("Accuracy: ",accuracy_score(Y_test, Y_pred2))
