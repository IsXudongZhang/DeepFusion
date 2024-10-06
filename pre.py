from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc

from torch.autograd import Variable
from torch.utils import data

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)
from argparse import ArgumentParser

from stream import BIN_Data_Encoder

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='MolTrans Testing.')
parser.add_argument('-b', '--batch-size', default=32, type=int,metavar='N')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')

def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label, d_sim, p_sim) in enumerate(tqdm(data_generator)):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda(), d_sim.float().cuda(), p_sim.float().cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).float()).cuda()

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    print("AUROC:" + str(auc_k))
    print("AUPRC: " + str(average_precision_score(y_label, y_pred)))

    cm1 = confusion_matrix(y_label, y_pred_s)
    print('Confusion Matrix : \n', cm1)
    print('Recall : ', recall_score(y_label, y_pred_s))
    print('Precision : ', precision_score(y_label, y_pred_s))

    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                              outputs), y_pred, loss.item()

def main():
    args = parser.parse_args()

    loss_history = []

    model = torch.load('./save_model/model.pth')

    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.workers,
              'drop_last': True}

    df_test = pd.read_csv('./dataset/DAVIS/test.csv')
    sim_path = './dataset/DAVIS/test/'
    testing_set = BIN_Data_Encoder(df_test.index.values, df_test.Label.values, df_test, sim_path)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Test ---')
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, f1, logits, loss = test(testing_generator, model)
            print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1) + ' , Test loss: ' + str(loss))
    except:
        print('Test failed')
    return model, loss_history



model, loss_history = main()

