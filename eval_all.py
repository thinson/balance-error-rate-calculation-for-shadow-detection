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
pre_path = r'C:\Users\kang_\Desktop\label'
label_path = r'C:\Users\kang_\Desktop\output'
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
    #TP
    tp_mat = torch.eq(pred_p,y_actual)
    TP = float(tp_mat.sum())

    #FN
    fn_mat = torch.eq(pred_n, y_actual)
    FN = float(fn_mat.sum())

    # FP
    fp_mat = torch.ne(y_actual, pred_p)
    FP = float(fp_mat.sum())

    # TN
    fn_mat = torch.ne(y_actual, pred_n)
    TN = float(fn_mat.sum())



    #print(TP,TN,FP,FN)
    #tot=TP+TN+FP+FN
    #print(tot)
    pos = TP+FN
    neg = TN+FP

    #print(pos,neg)

    #print(TP/pos)
    #print(TN/neg)
    '''
    if(pos!=0 and neg!=0):
        BAC = (.5 * ((TP / pos) + (TN / neg)))
    elif(neg==0):
        BAC = (.5*(TP/pos))
    elif(pos==0):
        BAC = (.5 * (TN / neg))
    else:
        BAC = .5
    
    # print('tp:%d tn:%d fp:%d fn:%d' % (TP, TN, FP, FN))
    accuracy = (TP+TN)/(pos+neg)*100
    BER=(1-BAC)*100
    '''
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

    print("name:%s , ber:%f,  accuracy:%f" % (name, global_ber,accuracy))
    print(difficult)


