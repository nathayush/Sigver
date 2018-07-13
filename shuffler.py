import csv, re
from glob import glob
from random import shuffle

def shuffleData():
    myData = []
    files = glob("data/images/*.PNG")
    for i in files:
        image = i[12:]
        target = i[12:].split('-')[0]
        forgery = 1 if image[:-4][-1]=='f' else 0
        myData.append([image, target, str(forgery)])

    dataFile = open('data/dataset.csv', 'w', newline='')
    with dataFile:
        writer = csv.writer(dataFile)
        writer.writerows([["image", "target", "forgery"]])
        writer.writerows(myData)

    shuffle(myData)
    trainData = myData[:int(len(myData) * .75)]
    testData = myData[int(len(myData) * .75):]

    trainFile = open('data/train_data.csv', 'w', newline='')
    with trainFile:
        writer = csv.writer(trainFile)
        writer.writerows([["image", "target", "forgery"]])
        writer.writerows(trainData)

    testFile = open('data/test_data.csv', 'w', newline='')
    with testFile:
        writer = csv.writer(testFile)
        writer.writerows([["image", "target", "forgery"]])
        writer.writerows(testData)
