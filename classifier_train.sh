
mpiexec -n 8 python classifier_train.py \
--log_path noised_classifier/logs \
--data_dir /data/QuickDraw_SketchRNN/img_64/train \
--val_data_dir /data/QuickDraw_SketchRNN/img_64/test \
--noised True \
--iterations 1000000 \
--anneal_lr True \
--batch_size 64 \
--lr 3e-4 \
--save_interval 5000 \
--log_interval 5000 \
--eval_interval 10000 \
--weight_decay 0.05 \
--step_size 25000 \
--classifier_attention_resolutions 32,16,8 \
--classifier_pool attention 

