import argparse
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

import cv2
import time
import numpy as np
import pandas as pd

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,10000).__str__()

def position_info(filename):
    return pd.read_csv(filename, usecols=[0, 4, 5], names=['a', 'b', 'c']).values.tolist()

def get_partition(img,rows,cols, width, height, size):
    #img = cv2.imread(src_img)
    #rows, cols = img.shape[0], img.shape[1]
    if size == 224:
        a = b = 112
    else:
        a = 149
        b = 150
    x_left = width - a
    y_left = height - b
    x_right = width + b
    y_right = height + a
    if x_left >=0 and y_left >=0 and x_right <= cols and y_right <= rows:
        return img[x_left: x_right, y_left:y_right]
    elif x_left < 0 or y_left < 0:
        return img[0: x_right, 0: y_right]
    elif x_right > cols or y_right > rows:
        return img[x_left: cols, y_left: rows]


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="151673")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    # args.cuda = torch.cuda.is_available()
    print(args)
    list = []
    position = position_info(f"./{args.name}/tissue_positions_list_{args.name}.csv")
    # print(position)
    start = time.time()
    size = args.size
    img = cv2.imread(f"./{args.name}/{args.name}.jpg")
    rows, cols = img.shape[0], img.shape[1]
    #os.makedirs(f'./{args.name}/img_{args.name}_{args.size}')
    os.makedirs(f'./test_time/img_{args.name}_{args.size}')
    print("mkdir sucessful!")
    for i in range(len(position)):
        img_part = get_partition(img,rows,cols, position[i][1], position[i][2], size)
        # cut_path = 'data/img/example_img_151673' + '/' + position[i][0] + '.jpg'  # .jpg图片的保存路径
        # cv2.imwrite(cut_path, img_part)

        img_part_npy = np.array(img_part)
        #cut_path_npy = f'./{args.name}/img_{args.name}_{args.size}' + '/' + position[i][0] + '.npy'  # .jpg图片的保存路径
        cut_path_npy = f'./test_time/img_{args.name}_{args.size}' + '/' + position[i][0] + '.npy'
        np.save(cut_path_npy,img_part_npy)
        #
        print(position[i][0]," was saved!!")
    
    end = time.time()
    print("run time: "+ str(end-start) + " seconds")




