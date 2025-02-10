import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

from tqdm import tqdm, trange


import torch
from safetensors.torch import load_file, save_file

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import StableDiffusion3Pipeline

from inference_configs import parse_args

def main():
    opt = parse_args()

    model_id = opt.pretrained_model_name_or_path
    if 'stable-diffusion-3.5-medium' not in model_id:
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        if 'stable-diffusion-xl-base-1.0' in model_id:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32
            )
    else: 
        pipe = StableDiffusion3Pipeline.from_pretrained(
           model_id,  torch_dtype=torch.float16)
        

    
    if opt.dora_ckpt_dir is not None:
        # load attention processors
        pipe.load_lora_weights(opt.dora_ckpt_dir)
        print('load DoRA checkpoint: done!')


    pipe = pipe.to("cuda")


    print(f"reading prompts from {opt.prompt_file_name}")
    with open(opt.prompt_file_name, "r") as f:
        data = f.read().splitlines()
        data = [d.strip("\n").split("\t")[0] for d in data] 
     

    # create output folder
    outpath = opt.outdir 
    sample_path = os.path.join(outpath, f"samples")
    os.makedirs(sample_path, exist_ok=True)
   
    # run inference
    with torch.no_grad():
        for prompt in tqdm(data, desc="data"):
      
            print(f'prompt {prompt}, seed {opt.seed}, save to {sample_path}')

            generator = torch.Generator(device="cuda").manual_seed(opt.seed)
            images = pipe(prompt=prompt,
                         num_inference_steps=opt.num_inference_steps, 
                         num_images_per_prompt = opt.num_images_per_prompt,
                         guidance_scale=opt.guidance_scale,
                         generator=generator,
                        ).images
        
            print('len(images)', len(images))
            for image in images:
                base_count = len(os.listdir(sample_path))
                image.save(os.path.join(sample_path, f"{prompt}_{base_count:06}.png"))
        

if __name__ == "__main__":
    main()
