import os
from ensemble import *
import pickle
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from feature import *
import numpy as np
def load_image(FACE,max_num = 200):
    featurelist = []
    #如果找到了已生成的pkl文件就返回
    if FACE == True:
        if os.path.exists("./face.pkl"):
            with open("./face.pkl" , "rb+") as f:
                featurelist = pickle.load(f)
            return featurelist
    elif FACE == False:
        if os.path.exists("./nonface.pkl"):
            with open("./nonface.pkl" , "rb+") as fn:
                featurelist = pickle.load(fn)
            return featurelist
    
    if FACE == True:
        dirPath =  "./datasets/original/face/"
    else:
        dirPath =  "./datasets/original/nonface/"
    
    dirlist = os.listdir(dirPath)
    count = 0
    for file in dirlist:
        if count < max_num:
            img = Image.open(dirPath + file )
            img = img.resize((24,24))
            img = img.convert('L')
            img_array = np.array(img)
            NPD = NPDFeature(img_array)
            NPD_feature = NPD.extract()
            featurelist.append(NPD_feature)
            count += 1
            print(count,NPD_feature.shape)
    
    if FACE == True:
        faceFile = open("face.pkl","wb")
        pickle.dump(featurelist,faceFile)
    else:
        nonfaceFile = open("nonface.pkl","wb")
        pickle.dump(featurelist,nonfaceFile )
    return featurelist

if __name__ == "__main__":
    # write your code here
    
    size = 500
    faceList = load_image(True , max_num = size)
    nonfaceList = load_image(False,max_num =size)

    labelList = [1] * size + [-1] * size
    X = faceList + nonfaceList
    print(len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, labelList, test_size=0.25)
    X_train = mat(X_train)
    X_test = mat(X_test)
    y_train = array(y_train)
    y_test = array(y_test)
    print(X_train.shape,y_train.shape)

    adaBoostClassifier = AdaBoostClassifier(weak_classifier = DecisionTreeClassifier ,
                                        n_weakers_limit = 25)
    adaBoostClassifier.fit(X_train,y_train)
    pred = adaBoostClassifier.predict(X_test,y_test)
    #write the report.txt
    target_names = ['nonface', 'face'] 
    report = classification_report(y_test, pred, target_names=target_names)
    print(report)
    with open("./report.txt","w") as f:
        f.write(report)

