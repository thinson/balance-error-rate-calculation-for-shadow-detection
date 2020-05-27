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
    if(pos!=0 and neg!=0):
        BAC = (.5 * ((TP / pos) + (TN / neg)))
    elif(neg==0):
        BAC = (.5*(TP/pos))
    elif(pos==0):
        BAC = (.5 * (TN / neg))
    else:
        BAC = .5
    print('tp:%d tn:%d fp:%d fn:%d' % (TP, TN, FP, FN))
    BER=(1-BAC)
    return BER

if __name__ == "__main__":
    # get img file in a list
    img_list = os.listdir(pre_path)
    print(img_list)
    average_ber = 0.0
    sum = 0.0
    for i,name in enumerate(img_list):
        if name.endswith('.png'):
            predict = cv2.imread(os.path.join(pre_path, name),cv2.IMREAD_GRAYSCALE)
            #print(predict)
            label = cv2.imread(os.path.join(label_path, name),cv2.IMREAD_GRAYSCALE)
            #print(label)
            score = BER(torch.from_numpy(label).float(), torch.from_numpy(predict).float())
            sum = sum + score
            average_ber = sum/(i+1)
            print("name:%s , ber:%f, average_ber:%f" % (name, score,average_ber))

