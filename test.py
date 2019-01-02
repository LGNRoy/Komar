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

(train_seq, valid_seq), (mean, std) = process_csv(filename="steering.csv", val=5)

test_seq = test_csv("exampleSubmissionFinal.csv")

print('--------------------train_seq--------------------------')
print(train_seq[:3])
print('---------------------test_seq--------------------------')
print(test_seq[:3])








