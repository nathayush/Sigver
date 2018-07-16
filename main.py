import trainer, dataset, shuffler
from sklearn import preprocessing

def main():
    num_classes = 21
    name_encoder = preprocessing.LabelBinarizer()

    shuffler.shuffleData() # shuffle dataset
    trainData = dataset.SigData("data/train_data.csv", name_encoder)
    testData = dataset.SigData("data/test_data.csv", name_encoder)
    print("created datasets.")
    mytrainer = trainer.Trainer(trainData, testData, num_users=num_classes) # number of people in the training set
    print("training.")
    mytrainer.train(num_epochs=60)

if __name__ == '__main__':
    main()
