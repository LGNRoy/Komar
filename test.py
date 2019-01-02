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

test_seq = read_csv("exampleSubmissionFinal.csv")

(train_seq, valid_seq), (mean, std) = process_csv(filename="steering.csv", val=5)

batch_generator = BatchGenerator(sequence=valid_seq, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
feed_inputs, feed_targets = batch_generator.next()

print(feed_inputs.shape)
print(feed_targets.shape)


# filename="steering.csv"
# val=5

# process_csv(filename,val)







