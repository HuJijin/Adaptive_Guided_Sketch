mpiexec -n 8 python diffusion_train.py \
--log_path duffusion/logs \
--data_dir /data/QuickDraw_SketchRNN/img_64/train \
--lr_anneal_steps 10000000 \
--batch_size 64 \
--lr 1e-4 \
--save_interval 8000 \
--log_interval 8000 \
--step_size 100000000 \
--channel_mult 1,2,3,4 \
--attention_resolutions 32,16,8 \
--dropout 0.1 

