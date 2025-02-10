import argparse

def parse_args_and_update_config(jupyter=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help='path to pretrained sd model',
    )
    parser.add_argument(
        "--dora_ckpt_dir", 
        type=str,  
        help='path to dora checkpoint folder, for round > 1',
    )


    parser.add_argument(
        "--prompt_file_name",
        type=str,
        default= '../T2I-CompBench/examples/dataset/color_train.txt',
        help='path to finetuning prompts txt file',
    )


    parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default= 50,
        help='number of candidate images',
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help='number of denoising timesteps',
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help='classifier free guidance scale',
    )
    

    parser.add_argument(
        "--outdir", 
        type=str,
        default='../MITUNE_data/color', 
    )

  
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()

    return args