from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
import torch
import datetime
import os
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def null_safety(images, **kwargs):
	return images, False


def make_pipline(model_path: str):
    # モデルの読み込み
    pipe = StableDiffusionPipeline.from_ckpt(
        model_path,
    )
    # negative prompt
    pipe.load_textual_inversion("./models/embeddings/EasyNegativeV2.safetensors", weight_name="EasyNegativeV2.safetensors", token="EasyNegativeV2")
    pipe.load_textual_inversion("./models/embeddings/negative_hand-neg.pt", weight_name="negative_hand-neg.pt", token="negative_hand-neg")
    # スケジューラーの設定
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # VAEの設定
    pipe.vae = AutoencoderKL.from_pretrained("./models/counterfeit_vae")
    # 黒塗り画像を出力しないようにする
    pipe.safety_checker = null_safety
    # cuda用
    pipe.enable_xformers_memory_efficient_attention()

    return pipe

def main(args):
    pipe = make_pipline("./models/checkpoints/Counterfeit-V3.0.safetensors")
    pipe = pipe.to("cuda")

    prompt = "(masterpiece, best quality),(((from below, depth of field, dutch angle, green lighting))), floating hair, 1girl, solo, formal, hand in pocket, suit, black gloves, building, looking at viewer, black necktie, fingerless gloves, white shirt, city, outdoors, black jacket, belt, black pants, collared shirt, brown eyes, standing, long sleeves, grey hair, cityscape, open jacket, cowboy shot, skyscraper, black suit, night, pant suit, very long hair"
    negative_prompt = "EasyNegativeV2, negative_hand-neg,(worst quality:1.4), (low quality:1.4), (monochrome:1.1),text,watermark"

    generator = torch.Generator(device="cuda").manual_seed(0)

    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=25,
        generator=generator,
        guidance_scale=10,
        # width=512,
        # height=512,
        ).images[0]

    image.save(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default=f"./save/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png")
    args = parser.parse_args()
    main(args) 