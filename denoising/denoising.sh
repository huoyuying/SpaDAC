#! /bin/bash

python denoising.py \
--exp_array './151673_cor/151673_rawexp.csv' \
--pos_adj './151673_cor/151673_adj.csv' \
--pos_dis './151673_cor/151673_dis.csv' \
--image_adj './151673_cor/adj_151673_inception_v3_cos.csv' \
--image_dis './151673_cor/cor_151673_inception_v3_cos.csv' \
--destination './151673_cor/output_inception_v3_cos.csv' \
