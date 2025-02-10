import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_file_name",
        type=str,
        help="load test prompts from this file",
    )
    parser.add_argument(
        "--outdir",
        type=str, 
        help="dir to write results to",
        default="../MITUNE_test"
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help='path to stable-diffusion-2-1-base',
        default="stabilityai/stable-diffusion-2-1-base", 
    )
    parser.add_argument(
        "--dora_ckpt_dir",  type=str, )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50, 
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=10, 
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5, 
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, 
    )

    args = parser.parse_args()
    return args

