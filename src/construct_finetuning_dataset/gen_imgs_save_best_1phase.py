import os
import torch

from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers import AutoencoderTiny
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipeline
from datasets import load_dataset

from image_retrieval_configs import parse_args_and_update_config
from pipeline_gen_mi_1phase import MISelectionPipeline, MISelectionXLPipeline
from pipeline_gen_mi_1phase_sd3 import StableDiffusion3Pipeline_MI


def main():
    if 'load config':
        config = parse_args_and_update_config()
   
    if 'load sd & dora':
        model_id = config.pretrained_model_name_or_path

        if 'stable-diffusion-3.5-medium' not in model_id:
            scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        if 'spright' in model_id:
            pipe = MISelectionPipeline.from_pretrained(model_id, 
                scheduler=scheduler,  
                torch_dtype=torch.float32,
                use_safetensors=True,
                token=os.getenv("HF_TOKEN"),
            )
        elif 'stable-diffusion-xl-base-1.0' in model_id:
            pipe = MISelectionXLPipeline.from_pretrained(model_id, 
                scheduler=scheduler,  
                torch_dtype=torch.float32,
                use_safetensors=True,
            )

        elif 'stable-diffusion-3.5-medium' in model_id:
            pipe = StableDiffusion3Pipeline_MI.from_pretrained(model_id, 
                torch_dtype=torch.float32,
                use_safetensors=True,
            )
        else:
            pipe = MISelectionPipeline.from_pretrained(model_id, 
                scheduler=scheduler,  
                torch_dtype=torch.float32,
                use_safetensors=True,
            )


        if config.dora_ckpt_dir:

            # state_dict, network_alphas = pipe.lora_state_dict(
            #     config.dora_ckpt_dir,
            #     unet_config=pipe.unet.config,
            # )

            # if "check dora state_dict":
            #     is_correct_format = all((("lora" in key) or ("dora_scale" in key)) for key in state_dict.keys())
            #     if not is_correct_format:
            #         raise ValueError("Invalid DoRA checkpoint.")
        
            # StableDiffusionLoraLoaderMixin.load_lora_into_unet(
            #     state_dict = state_dict,
            #     network_alphas = network_alphas, 
            #     unet = pipe.unet,
            # )

            # load attention processors
            pipe.load_lora_weights(config.dora_ckpt_dir, 
                                   wforeight_name='pytorch_lora_weights.safetensors')

            print('load DoRA weights done!')
        else:
            print('no   DoRA weights loaded!')


        pipe = pipe.to("cuda")
        # Combine attention projection matrices.
        # pipe.fuse_qkv_projections()
        
        # Initialize the weights in MI formula
        pipe.mi_weight_init()

    if 'load prompts':
        assert config.prompt_file_name is not None
        
        with open(config.prompt_file_name, "r") as f:
            prompts = f.read().splitlines()
            prompts = [d.strip("\n").split("\t")[0] for d in prompts] 
            
        print(f"prompts from {config.prompt_file_name}:  len={len(prompts)}") 

    
    if "makedir":
        data_outdir = config.outdir
        os.makedirs(data_outdir, exist_ok=True)


    seed = config.seed
    num_images_per_prompt = config.num_images_per_prompt
    num_inference_steps   = config.num_inference_steps
    guidance_scale = config.guidance_scale

    for prompt in prompts:
        if (os.path.isfile(
                os.path.join(data_outdir, f"{prompt}.png") if ('xl' not in model_id
        )  else os.path.join(data_outdir, f"{prompt[:200]}_prompt_embeds.pt")
            )):
            print(prompt)

        else:
            with torch.no_grad(): # , torch.cuda.amp.autocast() TODO: don't use autocast for xl

                print(f'prompt {prompt}, seed {seed}, save to {data_outdir}')
                generator = torch.Generator(device="cuda").manual_seed(seed)

     
                prompt_embeds, pooled_prompt_embeds, latents, images = pipe(
                    prompt = prompt, 
                    num_images_per_prompt = num_images_per_prompt,
                    num_inference_steps=num_inference_steps, 
                    generator=generator,
                    guidance_scale = guidance_scale,

                    return_dict = False
                )

                image = images[0] # .images


                os.makedirs(os.path.join(data_outdir, f"{prompt}_{seed}/"), exist_ok=True)
                
                torch.save(prompt_embeds, 
                            os.path.join(data_outdir, f"{prompt}_{seed}/prompt_embeds.pt"))
                if pooled_prompt_embeds is not None:
                    torch.save(pooled_prompt_embeds, 
                            os.path.join(data_outdir, f"{prompt}_{seed}/pooled_prompt_embeds.pt"))
                torch.save(latents, 
                            os.path.join(data_outdir, f"{prompt}_{seed}/latents.pt"))
                image.save(os.path.join(data_outdir, f"{prompt}_{seed}/image.png"))


                

if __name__ == "__main__":
    main()
