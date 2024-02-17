import config
import csv
import pandas as pd
import torch
import numpy as np

class TrainingStatistics:
    def __init__(self, record_path):
        self.record_path = record_path

    def save(self, *args):
        df = pd.DataFrame(args).transpose()
        df.to_csv(self.record_path, index=False, header=False)

    def load(self):
        return pd.read_csv(self.record_path, header=None).values.tolist()
    
class ModelStats:
    def save_checkpoint(epoch, model, optimizer, best_acc, path):
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        }
        torch.save(checkpoint, path)

    def load_model(path, epoch, model, optimizer, best_acc, device):
        checkpoint = torch.load(path, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_acc = checkpoint['best_acc']
        return epoch, model, optimizer, best_acc

# sample
'''
stats = TrainingStatistics(config.RECORD_PATH)
stats.save(trainloss, validloss, trainPCC, validPCC, trainiou, valiou, trainf1, valf1)
trainloss, validloss, trainPCC, validPCC, trainiou, valiou, trainf1, valf1 = stats.load()
model, optimizer = load_model('model_path.pth', model, optimizer, device)
'''

'''==============================================================================================================='''

'''舊版 for train_mito_nuclei'''

def savestatis(trainloss, validloss, trainPCC, validPCC, trainiou, valiou, trainf1, valf1): #trainr, valr):
    with open(config.RECORD_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(trainloss)
        w.writerow(validloss)
        w.writerow(trainPCC)
        w.writerow(validPCC)
        w.writerow(trainiou)
        w.writerow(valiou)
        w.writerow(trainf1)
        w.writerow(valf1)
        

def load_(root):
    Data = pd.read_csv(root, delimiter= ',', encoding= 'utf-8', header= None)
    data = Data.to_numpy()
    trainloss = data[0].tolist()
    validloss = data[1].tolist()
    trainPCC = data[2].tolist()
    validPCC = data[3].tolist()
    trainiou = data[4].tolist()
    valiou = data[5].tolist()
    trainf1 = data[6].tolist()
    valf1 = data[7].tolist()
    
    
    #trainloss = []
    #validloss = []
    #trainPCC = []
    #validPCC = []
    #trainiou = []
    #valiou =[]
    #trainf1 = []
    #valf1 = []
    #for i in range(0, np.size(data, axis= 1)):
    #    trainloss.data[0].tolist()
    #    validloss.append(data[1][i])
    #    trainPCC.append(data[2][i])
    #    validPCC.append(data[3][i])
    #    trainiou.append(data[4][i])   
    #    valiou.append(data[5][i])
    #    trainf1.append(data[6][i])
    #    valf1.append(data[7][i])
    
    return trainloss, validloss, trainPCC, validPCC, trainiou, valiou, trainf1, valf1

def savestatis(trainloss, validloss, trainPCC, validPCC, trainPSNR, valPSNR ,trainSSIM, valSSIM): #trainr, valr):
    with open(config.RECORD_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(trainloss)
        w.writerow(validloss)
        w.writerow(trainPCC)
        w.writerow(validPCC)
        w.writerow(trainPSNR)
        w.writerow(valPSNR)
        w.writerow(trainSSIM)
        w.writerow(valSSIM)
        

def load_(root):
    Data = pd.read_csv(root, delimiter= ',', encoding= 'utf-8', header= None)
    data = Data.to_numpy()
    trainloss = []
    validloss = []
    trainPCC = []
    validPCC = []
    trainPSNR = []
    valPSNR = []
    trainSSIM = []
    valSSIM = []
    for i in range(0, np.size(data, axis= 1)):
        trainloss.append(data[0][i])
        validloss.append(data[1][i])
        trainPCC.append(data[2][i])
        validPCC.append(data[3][i])
        trainPSNR.append(data[4][i])
        valPSNR.append(data[5][i])
        trainSSIM.append(data[6][i])
        valSSIM.append(data[7][i])
    return trainloss, validloss, trainPCC, validPCC,trainPSNR, valPSNR, trainSSIM, valSSIM

def loadmodel(root, model, opt, device):
    checkpoint = torch.load(root, map_location= device)
    model_state, optimizer_state = checkpoint["model"], checkpoint["optimizer"]
    model.load_state_dict(model_state)
    opt.load_state_dict(optimizer_state)
    return model, opt
    