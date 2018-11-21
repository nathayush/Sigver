import trainer, dataset
from sklearn import preprocessing
from sklearn import svm


def main():
    ### PHASE 1 ###

    num_classes = 21

    trainData = dataset.CNNData("data/cnn/train")
    print("created training set.")
    testData = dataset.CNNData("data/cnn/test")
    print("created test set.")

    mytrainer = trainer.Trainer(trainData, testData, num_users=49) # number of people in the training set

    mytrainer.clear_model_checkpoints()

    print("training.")
    mytrainer.train(num_epochs=60)
    exit()


    # Load checkpoint as trained weights for CNN

    ### PHASE 2 ###

    # new_dataset for a user
    # features = mytrainer.model.get_feature_vectors(new_dataset)
    forg_classifier = svm.svc(class_weight='balanced')
    forg_classifier_rbf = svm.svc(kernel='rbf', class_weight='balanced')
    # forg_classifier.fit(features, labels)
    # forg_classifier.predict(...)

if __name__ == '__main__':
    main()
