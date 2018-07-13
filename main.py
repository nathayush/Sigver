import trainer, dataset, shuffler

def main():
    shuffler.shuffleData() # shuffle dataset
    trainData = dataset.SigData("data/train_data.csv")
    testData = dataset.SigData("data/test_data.csv")
    print("created datasets.")
    mytrainer = trainer.Trainer(trainData, testData, num_users=21) # number of people in the training set
    print("training.")
    mytrainer.train()

if __name__ == '__main__':
    main()
