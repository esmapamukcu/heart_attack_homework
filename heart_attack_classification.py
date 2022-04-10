
import models_classification as mc


def main():
    x,y = mc.UserOperations.getDocName()
    testSize=mc.UserOperations.getTestsize()
    x_train, x_test, y_train, y_test = mc.trainTestsplit(x, y,testSize)
    X_train, X_test = mc.standardScaler(x_train, x_test)
    prediction = mc.UserOperations.dataForprediction()
    choice = mc.UserOperations.choice(X_train, X_test, y_train, y_test, prediction)
    
    


if __name__ == "__main__":
    main()    
    
    
    
    
        












