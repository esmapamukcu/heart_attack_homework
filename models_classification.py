
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import heart_attack_classification as main
import sys

class UserOperations():
    def choice(X_train,X_test, y_train,y_test, prediction):
            print("Hangi sınıflandırma algoritmasını kullanmak istiyorsunuz?\n1- Lojistik Regression\n2- K-En Yakın Komşu\n3- SVC\n4-Naive Bayes\n5- Decision Tree\n6- Random Forest")
            choice = input("Lütfen seçiminizi giriniz: ")
            try:
                choice = int(choice)
            except ValueError:
                print("Lütfen bir sayı değeri giriniz!")
            if choice == 1:
                ModelsOfClassification.logisticReg(X_train,X_test, y_train,y_test, prediction)
            elif choice == 2:
                ModelsOfClassification.knnClassifier(X_train,X_test, y_train,y_test, prediction)
            elif choice == 3:
                ModelsOfClassification.svc(X_train,X_test, y_train,y_test,prediction)
            elif choice == 4:
                ModelsOfClassification.naiveBayes(X_train,X_test, y_train,y_test, prediction)
            elif choice == 5:
                ModelsOfClassification.decisionTree(X_train,X_test, y_train,y_test, prediction)
            elif choice == 6:
                ModelsOfClassification.randomForest(X_train,X_test, y_train,y_test, prediction)
                
                
    def dataForprediction():
        prediction = [input("Tahmin için veri girişi yapınız (63, 1, 3, 145, 233, 1):")]
        
        return prediction
    
    def getDocName():                          
        while True:
            doc_name=input("Csv dosyasının adını giriniz (örn:heart.csv) :")
            try:
                datas=pd.read_csv(doc_name)
                break                           
            except FileNotFoundError:
                print(doc_name ," isimli dosya bulunamadı. Lütfen tekrar giriş yapınız. ")
                continue    
            
        #x_data=input("Lütfen x(bağımsız) değişkenleri belirleyiniz (örn: ':,0:6' ):")
        #y_data=input("Lütfen y(bağımlı) değişkenleri belirleyiniz ( örn: ':,13:14' ):")
        x=datas.iloc[:,0:6].values
        y=datas.iloc[:,13:14].values
        return x,y
    
    
    def getTestsize():
        testSize=input("Lütfen verinin eğitim-test kümelerine ayrılması için 0 ve 1 arası bir test değeri giriniz (örn:0.20):")
        """
        try:
            testSize = float(testSize) and  testSize>0 and testSize<1
            
        except ValueError:
            print("Lütfen 0 ve 1 arası değer giriniz ve ondalık sayılar için nokta kullanınız.")
            sys.exit()
        except TypeError:
            print("Lütfen 0 ve 1 arası değer giriniz ve ondalık sayılar için nokta kullanınız.")
            sys.exit()            
          
        return testSize
        """

class ModelsOfClassification():

    # Logistic Regression
    def logisticReg(X_train, X_test, y_train,y_test,prediction):
        logr = LogisticRegression(random_state=0)
        logr.fit(X_train, y_train)  
    
        y_pred = logr.predict(X_test)  
        print("Logistic Regression: ")
        print("------------------------------")
        Metrics.accuracyScore(y_test, y_pred)
        print("------------------------------")
        Metrics.confusionMatrix(y_test, y_pred)
        print("------------------------------")
        print("Prediction result :")
        print((logr.predict([prediction])))   #63, 1, 3, 145, 233, 1
    

    #K-NN 
    def knnClassifier(X_train, X_test, y_train,y_test,prediction):
        knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
        knn.fit(X_train, y_train)
    
        y_pred = knn.predict(X_test)
        
        print("K-NN Classifier: ")
        print("------------------------------")
        Metrics.accuracyScore(y_test, y_pred)
        print("------------------------------")
        Metrics.confusionMatrix(y_test, y_pred)
        print("------------------------------")
        print("Prediction result :")
        print((knn.predict([[prediction]])))

    #SVC (Support Vector SVM classifier)
    def svc(X_train, X_test, y_train,y_test,prediction):
        svc = SVC(kernel='poly')
        svc.fit(X_train, y_train)
    
        y_pred = svc.predict(X_test)
        print("Support Vector Classifier: ")
        print("------------------------------")
        Metrics.accuracyScore(y_test, y_pred)
        print("------------------------------")
        Metrics.confusionMatrix(y_test, y_pred)
        print("------------------------------")
        print("Prediction result :")
        print((svc.predict([[prediction]])))

    # Naive Bayes
    def naiveBayes(X_train, X_test, y_train,y_test,prediction):
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
    
        y_pred = gnb.predict(X_test)
        print("Naive Bayes: ")
        print("------------------------------")
        Metrics.accuracyScore(y_test, y_pred)
        print("------------------------------")
        Metrics.confusionMatrix(y_test, y_pred)
        print("------------------------------")
        print("Prediction result :")
        print((gnb.predict([[prediction]])))

    # Decision tree
    def decisionTree(X_train, X_test, y_train,y_test,prediction):
    
        dtc = DecisionTreeClassifier(criterion='entropy')
        
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        print("Decision Tree: ")
        print("------------------------------")
        Metrics.accuracyScore(y_test, y_pred)
        print("------------------------------")
        Metrics.confusionMatrix(y_test, y_pred)
        print("------------------------------")
        print("Prediction result :")
        print((dtc.predict([[prediction]])))

    # Random Forest
    def randomForest(X_train, X_test, y_train,y_test,prediction):
     
        rfc = RandomForestClassifier(n_estimators=75, criterion='entropy')
        rfc.fit(X_train, y_train)
        
        y_pred = rfc.predict(X_test)
        print("Random forest: ")
        print("------------------------------")
        Metrics.accuracyScore(y_test, y_pred)
        print("------------------------------")
        Metrics.confusionMatrix(y_test, y_pred)
        print("------------------------------")
        print("Prediction result :")
        print((rfc.predict([[prediction]])))
    
class Metrics():    
    def accuracyScore(y_test, y_pred):
        print("Accuracy score: ")
        ac_s = accuracy_score(y_test, y_pred)
        print(ac_s)
        
        
        
    def confusionMatrix(y_test, y_pred):
        print("Confusion Matrix: ")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        return cm
  
    
def trainTestsplit(x,y,testSize):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=testSize, random_state=0)
    return x_train, x_test, y_train, y_test


# Standardization
def standardScaler(x_train,x_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)
    X_test = sc.transform(x_test)
    return X_train, X_test   


 
    
    
    
    