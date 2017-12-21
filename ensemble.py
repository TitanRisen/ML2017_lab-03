import pickle

from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from numpy import *
class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
       
        #存放训练好的弱分类器和alpha
        self.weak_classifier_list = []
        self.alpha_list = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        m,n = X.shape
        Classifier = self.weak_classifier
        errorRate_list=[]
        #initialize D (weights)
        D = array(ones(m))
        D = D * (1/m)
        #最终估计分类用矩阵
        totalClassEst = array(zeros(m))
        for i in range(self.n_weakers_limit):
            #fit every weak Classifier and add its prediction to the list
            #use just 1 floor
            self.weak_classifier_list.append(Classifier(max_depth = 1 ))
            self.weak_classifier_list[i].fit(X,y,sample_weight = D)
            pred = self.weak_classifier_list[i].predict(X)
            # 1^-1=-2 , 1^1 = 0 , -1^-1 = 0
            #error_num = -(pred ^ y).sum()/2
            error_num = (pred != y).sum()
            #culcate the error rate
            error = error_num / m
            print(i," the sub error rate is ",error)
            alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
            # if predict is correct, then *exp(-alpha) to make the weights smaller ,else * exp(alpha)
            expon = multiply(-1*alpha*y, pred)
            D = multiply(D,exp(expon))
            D = D/D.sum()
            
            #add the alpha to the class Container
            self.alpha_list.append(alpha)
            
            #Calculate the total error of the training set
            totalClassEst += alpha * pred
            ClassEst = sign(totalClassEst)
            ClassEst_errors = (ClassEst != y)
            total_error_rate = ClassEst_errors.sum()/m
            print(" the total error rate is ",total_error_rate)
            errorRate_list.append(total_error_rate) 
            #if the training model has no error ,it can stop          
            if total_error_rate == 0:break
        
        #For convenience , I save the training model
        self.save(self.alpha_list , "./alpha_list.pkl")
        self.save(self.weak_classifier_list,"./weak_classifier_list.pkl")
        print("train success")
        #self.draw_curve(errorRate_list)
    

    def draw_curve(self,errorRate_list):
        iter_list = []
        for i in range(len(errorRate_list)):
            iter_list.append(i)
        plt.figure()

        plt.plot(iter_list,errorRate_list,linestyle = '-.',color = 'black',linewidth = 2.0, label = 'test')
        plt.xlabel('iter_times')  
        plt.ylabel('error_rate')
        plt.legend()
        plt.show()

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, y,threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        #load the model which has been trained
        weak_classifier_list = self.load("./weak_classifier_list.pkl")
        alpha_list = self.load('./alpha_list.pkl')
        m,n = X.shape
        ClassEst = 0
        totalClassEst = array(zeros(m))
        errorRate_list = []
        for i in range(len(weak_classifier_list)):
            #using the weak classifier and alpha stored.
            pred = weak_classifier_list[i].predict(X)
            totalClassEst += alpha_list[i] * pred
            ClassEst = sign(totalClassEst)
            ClassEst_errors = (ClassEst != y)
            total_error_rate = ClassEst_errors.sum()/m
            print(i," the total error rate in predicting the validation set is ",total_error_rate)
            errorRate_list.append(total_error_rate)
            #if the training model has 99% correct rate ,it can stop 
            if total_error_rate <= 0.01:break
        self.draw_curve(errorRate_list)
        return ClassEst

    @staticmethod
    def save(model, filename):
        with open(filename, "wb+") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
