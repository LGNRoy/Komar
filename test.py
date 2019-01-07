import tensorflow as tf
import numpy as np
import os
from Batch import *


SEQ_LEN = 10 
BATCH_SIZE = 4 
LEFT_CONTEXT = 5

CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3] # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS) # predict all features: steering angle, torque and vehicle speed

loacl_camera_path = "E:/udacity-baseline/HMB6/camera.csv"
loacl_steering_path = "E:/udacity-baseline/HMB6/steering.csv"

loacl_tests_path = "E:/udacity-baseline/testing-data/camera.csv"
loacl_result_path = "E:/udacity-baseline/testing-data/ch2_final_eval.csv"

# # 在能跑之前,暂且使用HMB6代替全部数据
(train_seq, valid_seq), (mean, std) = process_csv(loacl_camera_path, loacl_steering_path, val=5)
# # test数据集里面的steering是空的,暂且使用HMB3代替
# test_seq = test_csv("/home/sway007/git-repos/udacity-driving-reader/output/HMB3/steering.csv")


# print('--------------------train_seq--------------------------')
print(train_seq[:3])
# print('---------------------test_seq--------------------------')
# print(test_seq[:3])

# a = read_csv_dou(loacl_camera_path, loacl_steering_path)
# print(a[:3])

# def new_read_csv(filename):
#     with open(filename, 'r') as f:
#         lines = [ln.strip().split(",") for ln in f.readlines()][1:] 
#         return lines

# test = new_read_csv(loacl_tests_path)
# result = new_read_csv(loacl_result_path)

# print("test  : {}".format(len(test)))
# print("result: {}".format(len(result)))

# def pocess(test, result):
#     count = 0
#     for i in range(len(test)):
#         if(test[i][0] == result[i][0]):
#             count+=1
#     print(count)
#     return ""

# pocess(test, result)
