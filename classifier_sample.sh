# DDIM
SAMPLE_FLAGS="--log_path samples --use_ddim True --num_samples 16 --batch_size 16 --timestep_respacing ddim250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --classifier_pool attention --classifier_attention_resolutions 32,16,8 --model_path models/64x64_uncond_diffusion.pt --classifier_path models/64x64_classifier.pt --class_path datasets/quickdraw_345.txt"
python classifier_sample.py $MODEL_FLAGS --input_label "basket" --show True --seed 2023 $SAMPLE_FLAGS 




