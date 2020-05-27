import os, cv2
import numpy as np
import torch
'''
discription: for balance error rate calculation
author:wukang
date: 2020-05-27
history:
    version v1.0 
'''


# set prediction path and label path for eval
#pre_path = r'C:\Users\kang_\Desktop\label'
pre_path = '/home/wukang/Projects/DSC-PyTorch-master/results'
#label_path = r'C:\Users\kang_\Desktop\output'
label_path = '/home/wukang/Projects/DSC-PyTorch-master/SBU-shadow/SBU-Test/ShadowMasks'
#label_path = r'C:\Users\kang_\Desktop\label'


def BER(y_actual, y_hat):
    y_hat = y_hat.ge(128).float()
    y_actual = y_actual.ge(128).float()

    y_actual = y_actual.squeeze(1)
    y_hat = y_hat.squeeze(1)

    #output==1
    pred_p=y_hat.eq(1).float()
    #print(pred_p)
    #output==0
    pred_n = y_hat.eq(0).float()
    #print(pred_n)
    #total_true
    pre_positive = float(pred_p.sum())
    pre_negtive = float(pred_n.sum())

    # FN
    fn_mat = torch.gt(y_actual, pred_p)
    FN = float(fn_mat.sum())

    # FP
    fp_mat = torch.gt(pred_p, y_actual)
    FP = float(fp_mat.sum())

    TP = pre_positive - FP
    TN = pre_negtive - FN


    return TP,TN,FP,FN

if __name__ == "__main__":
    # get img file in a list
    img_list = os.listdir(pre_path)
    print(img_list)
    sum_tp = 0.0
    sum_tn = 0.0
    sum_fp = 0.0
    sum_fn = 0.0
    difficult = []
    for i,name in enumerate(img_list):
        if name.endswith('.png'):
            predict = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            label = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            #print(label)
            TP, TN, FP, FN = BER(torch.from_numpy(label).float(), torch.from_numpy(predict).float())

            sum_tp = sum_tp + TP
            sum_tn = sum_tn + TN
            sum_fp = sum_fp + FP
            sum_fn = sum_fn + FN
    pos = sum_tp + sum_fn
    neg = sum_tn + sum_fp

    if (pos != 0 and neg != 0):
        BAC = (.5 * ((sum_tp / pos) + (sum_tn / neg)))
    elif (neg == 0):
        BAC = (.5 * (sum_tp / pos))
    elif (pos == 0):
        BAC = (.5 * (sum_tn / neg))
    else:
        BAC = .5
    accuracy = (sum_tp + sum_tn) / (pos + neg) * 100
    global_ber = (1 - BAC) * 100

    print(" ber:%f,  accuracy:%f" % ( global_ber,accuracy))
    #print(difficult)


