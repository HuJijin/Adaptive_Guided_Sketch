"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import time
import numpy as np
import torch as th
import torch.nn.functional as F
from math import ceil
from torchvision.utils import make_grid, save_image

from guided_diffusion.labels_to_names import label2index
from guided_diffusion.util import compute_top_k
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.log_path)
    
    # set random seed
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)
    
    with open(os.path.join(args.log_path, "config.txt"), "w") as f:
        config_dict = vars(args)
        for k,v in config_dict.items():
            f.write(f"{k} : {v}")
            f.write('\n')
        f.close()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)] 
            return th.autograd.grad(selected.sum(), x_in)[0]
        
    def classifier_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True) 
            logits = classifier(x_in, t)  
            probs = F.softmax(logits, dim=-1).detach() 
            tgt_class_prob = probs[:,y[0]] 
            tgt_class_prob = th.mean(tgt_class_prob).item()
            
            sorted_prbs, _ = th.sort(probs, dim=1, descending=True)
            diff_probs = sorted_prbs[:, 0] - sorted_prbs[:, 1]
            diff_probs = th.mean(diff_probs)
             
            acc_1 = compute_top_k(         
                    logits, y, k=1, reduction="none"
                ).cpu().numpy()

            return acc_1, diff_probs.item(), tgt_class_prob
        
    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.input_label:
            ones = th.ones(size=(args.batch_size,), dtype=th.int64, device=dist_util.dev())
            class_index = label2index(args.class_path, args.input_label)
            classes = ones * class_index
        else :
            classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
            
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            classifier_fn=classifier_fn,  # classifier_fn
            device=dist_util.dev(),
        )
        
        sample = sample.permute(0, 2, 3, 1)
        sample = th.clamp(sample, min=0, max=1)
        sample = sample * 255
        sample = sample.cpu().detach().numpy().astype('uint8')
        
        all_images.append(sample)
        
        logger.log(f"created {len(all_images) * args.batch_size} samples")


    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{args.seed}_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr)

    if args.show:
        samples = th.from_numpy(arr).float()
        samples = samples.permute(0, 3, 1, 2)
        samples = samples / 255.0
        for i in range(ceil(samples.shape[0]/64)):
            if i<5:
                imgs = samples[i*64:(i+1)*64]
                grid = make_grid(imgs)
                img_path = os.path.join(logger.get_dir(), f"samples_{i}.png")
                save_image(grid, img_path)

    # dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        log_path="",
        clip_denoised=True,
        num_samples=1000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=0.0,
        class_path="",
        input_label="",
        seed=0,
        show=True,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
