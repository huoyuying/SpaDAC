'''
Inference of inception-v3 model with pretrained parameters on ImageNet
'''
import time
import sys
import tensorflow.compat.v1 as tf
import argparse
# To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.disable_eager_execution()
import tensorflow_hub as hub
import numpy as np
import cv2
import pandas as pd


parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--name", type=str, default="151673")
parser.add_argument("--model", type=str, default="inception_v3")
parser.add_argument("--size", type=int, default=299)    
args = parser.parse_args()

if args.model == "inception_v3":
    args.size = 299
    module = hub.Module(f"./{args.model}")
elif args.model == "inception_resnet_v2":
    args.size = 299
    module = hub.load(f"./{args.model}")
elif args.model == "resnet50":
    args.size = 224
    module = hub.load(f"./{args.model}") 
else:
    args.size = 299
# Load saved inception-v3 model
# module = hub.Module(f"./{args.model}")

# images should be resized to 299x299
size = args.size
input_imgs = tf.placeholder(tf.float32, shape=[None, size, size, 3])
features = module(input_imgs)

# Provide the file indices
# This can be changed to image indices in strings or other formats
# spot_info = pd.read_csv(f'./spot_info_{args.name}.csv', header=0, index_col=None)
spot_info = pd.read_csv(f'./cut_image/{args.name}/tissue_positions_list_{args.name}.csv', header = None, index_col=None)
print(spot_info)
image_no = spot_info.shape[0]

start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    img_all = np.zeros([image_no, size, size, 3])

    # Load all images and combine them as a single matrix
    # This loop can be changed to match the provided image information
    for i in range(image_no):
        # Here, all images are stored in example_img and in *.npy format
        # if using image format, np.load() can be replaced by cv2.imread()
        # print(f"sys arhv2 : {sys.argv[2]}")
        # file_name = str(f'./cut_image/img_{args.name}_{args.size}' + spot_info.iloc[i, 2] + '.npy'
        # print(file_name)
        # temp = np.load(file_name)
        # temp = np.load(f'./cut_image/img_{args.name}_{args.size}/'+spot_info.iloc[i, 2] + '.npy')
        #temp = np.load(f'./cut_image/{args.name}/img_{args.name}_{args.size}/'+spot_info.iloc[i, 0] + '.npy')
        temp = np.load(f'./cut_image/test_time/img_{args.name}_{args.size}/'+spot_info.iloc[i, 0] + '.npy')
        temp2 = temp.astype(np.float32) / 255.0
        img_all[i, :, :, :] = temp2

    # Check if the image are loaded successfully.
    if (i == image_no - 1):
        print('+++Successfully load all images+++')
    else:
        print('+++Image patches missing+++')

    # Input combined image matrix to Inception-v3 and output last layer as deep feature
    fea = sess.run(features, feed_dict={input_imgs: img_all})
    print("step1 sucessful!!")

    # Save inferred image features
    #np.save(f'./cut_image/{args.name}/emb_{args.name}_{args.model}.npy', fea)
    np.save(f'./cut_image/test_time/emb_{args.name}_{args.model}.npy', fea)
    print("step2 sucessful!")
    end = time.time()
    print("run time: "+ str(end-start) + " seconds")

    # npy-->txt
    # read_npy = np.load(f'./emb_{args.name}_{args.model}.npy')
    # np.savetxt(f'./emb_{args.name}_{args.model}.csv',read_npy,delimiter=',')
    # print("run sucessful!!")
    
 

