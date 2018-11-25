import trainer, dataset
from sklearn import svm
from smrt.balance import smrt_balance
from smrt.autoencode import VariationalAutoEncoder
import torch

def main():
    ##################  PHASE 1  ##################
    # print("creating datasets.")
    # trainData = dataset.CNNData("data/train")
    # print("created training set.")
    # valData = dataset.CNNData("data/val")
    # print("created validation set.")
    #
    # X_train = trainData.x_data.view(1404, 150*220)
    # y_train = trainData.y_data
    # x_train = X_train[y_train.type(torch.ByteTensor)]
    #
    # print("training autoencoder.")
    # v_encoder = VariationalAutoEncoder(n_epochs=110, n_hidden=900, n_latent_factors=15,
    #                                learning_rate=1e-8, batch_size=256, display_step=10,
    #                                activation_function='sigmoid', verbose=2, l2_penalty=None,
    #                                random_state=42, early_stopping=True, dropout=0.4,
    #                                learning_function='sgd', clip=False)
    # v_encoder.fit(x_train)
    #
    # print("balancing dataset")
    # X_smrt, y_smrt = smrt_balance(X_train, y_train, n_hidden=900, n_latent_factors=10, random_state=42,
    #                           shuffle=False, balance_ratio=0.5, return_estimators=False,
    #                           prefit_estimators={1: v_encoder})
    # X_smrt = X_smrt.reshape(X_smrt.shape[0], 150, 220)
    # smrtData = dataset.SMRTData(X_smrt, y_smrt)
    #
    # mytrainer = trainer.Trainer(smrtData, valData)
    # mytrainer.clear_model_checkpoints()
    #
    # print("training.")
    # mytrainer.train(num_epochs=60)

    ##################  PHASE 2  ##################

    print("creating datasets.")
    trainData = dataset.CNNData("data/train")
    print("created training set.")
    valData = dataset.CNNData("data/val")
    print("created validation set.")

    mytrainer = trainer.Trainer(trainData, valData)

    checkpoint = torch.load("Signet_45.pkl")
    mytrainer.model.load_state_dict(checkpoint)

    print("creating testing set.")
    testData = dataset.CNNData("data/old/v3/val")
    print("testing the model.")
    mytrainer.test(testData)

if __name__ == '__main__':
    main()
