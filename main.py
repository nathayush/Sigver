import trainer, dataset, shuffler
from sklearn import preprocessing
from sklearn import svm


def main():
    num_classes = 21

    ### PHASE 1

    name_encoder = preprocessing.LabelBinarizer()

    shuffler.shuffleData() # shuffle dataset
    trainData = dataset.SigData("data/train_data.csv", name_encoder)
    testData = dataset.SigData("data/test_data.csv", name_encoder)
    print("created datasets.")
    mytrainer = trainer.Trainer(trainData, testData, num_users=num_classes) # number of people in the training set
    print("training.")
    mytrainer.train(num_epochs=60)

    exit()

    ### PHASE 2

    # new_dataset for a user
    # features = mytrainer.model.get_feature_vectors(new_dataset)
    forg_classifier = svm.svc(class_weight='balanced')
    forg_classifier_rbf = svm.svc(kernel='rbf', class_weight='balanced')
    # forg_classifier.fit(features, labels)
    # forg_classifier.predict(...)

if __name__ == '__main__':
    main()
