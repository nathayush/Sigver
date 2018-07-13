import trainer, dataset

def main():
    mydata = dataset.SigData()
    mytrainer = trainer.Trainer(mydata, num_users=21) # number of people in the training set
    mytrainer.train()

if __name__ == '__main__':
    main()
