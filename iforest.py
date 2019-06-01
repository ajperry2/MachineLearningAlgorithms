import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import time

class IsolationTreeEnsemble:
    '''
    A strong learner for anomaly detection
    
    This is a very fast model and does a very good job. I use this as a feature generator often,
    as this result is very useful to other models. 
    
    The Algorithm:
        We take our features and select a random feature and then random split point. We hypothesize
        that if a point is an outlier then it will tend to be isolated more often this way. So we score
        points based off of their tendency to be isolated quickly.
        
        The canonical paper can be found here:
        https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest
    
    Customization:
        For my purposes I usually have columns which I do not want considered as they "Water down"
        my results. I have added some preprocessing which has shown to improve results in the presence of
        noisy columns. This does add to training time, but it can easily by setting the 'improved' 
        parameter to False
    '''
    def __init__(self, sample_size, n_trees=10):
        '''
        Parameters:
        
        sample_size: A real number which impacts the scoring of points and the amount of data in 
                     training. The paper mentioned they found empirically that 256 is often enough
                     to get reliable results.
        
        n_trees: The number of weak learners to include. Higher numbers can lead lead to overfitting
                 and slower training. Only crank this number up for large data!
        '''
        self.trees = []
        self.n_trees = n_trees
        self.sample_size = np.array(sample_size)
        self.c = 2 * (np.log2(self.sample_size-1)+ 0.5772156649)-(2*(self.sample_size-1)/self.sample_size)

    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        #    Adjusting for noise
        # 1. Find six random split points for each column.
        # 2. Add up the minumum split size.
        # 3. Remove the columns with the largest sums.
        if improved:
            Y = X.T
            columns = np.array(range(Y.shape[0]))
            good_cols = []
            mins = None
            maxs = None
            mins = [column.min() for column in Y]
            maxs = [column.max() for column in Y]
            orig_col_len = len(columns)
            # This needs to be customized based on the number of columns in dataset
            if orig_col_len >20: 
                while len(columns)>0:
                    column_split_potential = []
                    #indeces of 3 random columns
                    cols = np.random.choice(columns,size=7)
                    
                    for col in cols:
                        column = Y[col]
                        sum_ = 0
                        splits= np.random.uniform(mins[col],maxs[col],size=40)

                        X1=[(column < split).sum() for split in splits]
                        X2=[(column >= split).sum() for split in splits]
                        least_partitions = [ min(x1,x2) for x1,x2 in zip(X1,X2)]

                        column_split_potential.append(sum(least_partitions))
                    # Keep minimum
                    good_cols.append(cols[np.argmin(column_split_potential)])
                    temp_max = cols[int(np.array(column_split_potential).argmax())]
                    columns = columns[columns!=cols[np.argmin(column_split_potential)]]

                    columns = columns[columns!=temp_max]
            else:
                while len(columns)>0:
                    column_split_potential = []
                    #indeces of 3 random columns
                    cols = np.random.choice(columns,size=3)

                    for col in cols:
                        column = Y[col]
                        sum_ = 0
                        splits= np.random.uniform(mins[col],maxs[col],size=40)

                        X1=[(column < split).sum() for split in splits]
                        X2=[(column >= split).sum() for split in splits]
                        least_partitions = [ min(x1,x2) for x1,x2 in zip(X1,X2)]

                        column_split_potential.append(sum(least_partitions))
                    # Keep minimum
                    good_cols.append(cols[np.argmin(column_split_potential)])
                    temp_max = cols[int(np.array(column_split_potential).argmax())]
                    columns = columns[columns!=cols[np.argmin(column_split_potential)]]

                    columns = columns[columns!=temp_max]
            #prune good cols
            if len(good_cols) > 10:

                sums_ = []

                for col in list(set(good_cols)):
                    column = Y[col]
                    splits= np.random.uniform(mins[col],maxs[col],size=20)

                    X1=[(column < split).sum() for split in splits]
                    X2=[(column >= split).sum() for split in splits]
                    least_partitions = [ min(x1,x2) for x1,x2 in zip(X1,X2)]
                    sums_.append(sum(least_partitions))
                sums_= np.array(list(set(sums_)))
                temp_max = np.array(list(set(good_cols)))[sums_.argsort()[:-2]]
                good_cols = temp_max
            good_cols = list(set(int(x) for x in good_cols))
            print(good_cols)
            Y= Y[good_cols]
            X=Y.T
        # Maximum height of a tree.
        height_limit = np.ceil(np.log2(self.sample_size))
        
        for tree_num in range(self.n_trees):
            # Sample data.
            np.random.seed(seed=2*tree_num)
            data_index = np.random.randint(low=0,high=X.shape[0],size=self.sample_size)
            sample = X[data_index,:]
            
            # Fit tree, initialize parameters, add to list
            new_tree = IsolationTree(sample,0,height_limit,self.sample_size)
            root = new_tree.fit(sample)
            new_tree.root = root
            new_tree.n_nodes = new_tree.root.n_nodes
            self.trees.append(new_tree)
        self.depth_matrix = np.zeros(shape=(X.shape[0],self.n_trees),dtype=float)
        
        return self


    def anomaly_score(self,X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        for tree_idx in range(self.n_trees):
            self.trees[tree_idx].find_depth(indeces=np.array(range(X.shape[0])),\
                                            X=X,\
                                            forrest=self,\
                                            tree_index=tree_idx,\
                                           node = self.trees[tree_idx].root)
        return np.power(2, -self.depth_matrix.mean(axis=1)/self.c)
    
    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        result = np.zeros(scores.shape)
        result[scores>threshold] = 1
        return result
    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        scores = anomaly_score(X)
        return predict_from_anomaly_scores(scores,threshold)
    
def find_TPR_threshold(y, scores, desired_TPR):
    for i in np.linspace(start=1,stop=0,num=1000):
        result = np.zeros(scores.shape)
        result[scores>i] = 1
        con_mat = confusion_matrix(y, result)
        TN, FP, FN, TP= con_mat.flat
        TP_Rate = TP / (TP + FN)
        FP_Rate = FP / (FP + TN)
        if TP_Rate > desired_TPR:
            return i, FP_Rate

class IsolationTree:
    def __init__(self, X:np.ndarray,current_height,height_limit,sample_size):
        self.height_limit = height_limit
        self.current_height = current_height
        self.sample_size=sample_size
        self.root = None
        self.n_nodes = None
    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.

        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        return self.build_tree(X,0)
        
    def build_tree(self,X,current_height=0):
        
        if X.shape[0] <=1 or current_height >= self.height_limit:
            return exNode(size = X.shape[0],depth=np.array(current_height))

        # Randomly select attribute index.
        q = np.random.randint(low=0,high=X.shape[1])

        # Randomly select split value.
        r = np.random.random()
        min_q = X[:,q].min()
        max_q = X[:,q].max()
        split = min_q + r * (max_q - min_q) 
        
        # Conditions of returning terminating node are split up so we do not need to compute max each time.
        if min_q == max_q : return exNode(size = X.shape[0],depth=np.array(current_height))

        # Split X  on q attribute.
        left_index = (X[:,q] < split )
        x1=X[left_index ]
        x2=X[np.invert(left_index)]            
        
        # Make children...
        parent = inNode(
            left=IsolationTree(x1,current_height+1,self.height_limit,self.sample_size)
                    .build_tree(x1,current_height+1),
            right=IsolationTree(x2,current_height+1,self.height_limit,self.sample_size)
                    .build_tree(x2,current_height+1),
            splitAtt=q,
            splitVal=split)
        return parent  


    def find_depth(self,indeces,X,forrest,tree_index,node):
        '''
        A helper function which returns the depth in a tree which a value appears.
        '''

        if not isinstance(node,exNode):
            #update relevant matrix
            X_left = X[:,node.splitAtt] < node.splitVal
            X_right = np.invert(X_left)
            #descend
            left = node.left
            self.find_depth(indeces[X_left],X[X_left],forrest,tree_index,left)
            right = node.right
            self.find_depth(indeces[X_right],X[X_right],forrest,tree_index,right)
        else:        #pass relevant X vals
            forrest.depth_matrix[indeces,tree_index] = node.depth
class inNode:
    def __init__(self,left,right,splitAtt,splitVal):
        self.right = right
        self.left = left
        self.splitAtt = splitAtt
        self.splitVal = splitVal
        self.n_nodes=0
        # Parameter n_nodes is passed up to the root to know how many nodes are in a tree.
        if isinstance(left,exNode) and isinstance(right,exNode) :
            self.n_nodes = 2 + 1
        elif isinstance(left,exNode) and isinstance(right,inNode) :
            self.n_nodes = 2 + right.n_nodes
        elif isinstance(left,inNode) and isinstance(right,exNode) :
            self.n_nodes = left.n_nodes + 2
        elif isinstance(left,inNode) and isinstance(right,inNode) :
            self.n_nodes = left.n_nodes + right.n_nodes + 1


class exNode:
    def __init__(self,size,depth):
        self.depth = depth
        self.size = size
        self.n_nodes = 1
        if size >2:
            self.c = 2 * (np.log2(size-1)+0.577215)-(2*(size-1)/size)
        else: # Must have at least one value
            self.c = 1