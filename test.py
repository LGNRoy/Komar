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



filename="steering.csv"
val=5

process_csv(filename,val)







