from fastapi import FastAPI, Response
import torch
import datetime
import os
import uvicorn
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

app = FastAPI()

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

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.get("/generate")
def generate(prompt: str, seed: int):

    # プロンプト
    prompt = "(masterpiece, best quality),(((from below, depth of field, dutch angle, green lighting))), floating hair, 1girl, solo, formal, hand in pocket, suit, black gloves, building, looking at viewer, black necktie, fingerless gloves, white shirt, city, outdoors, black jacket, belt, black pants, collared shirt, brown eyes, standing, long sleeves, grey hair, cityscape, open jacket, cowboy shot, skyscraper, black suit, night, pant suit, very long hair"
    negative_prompt = "EasyNegativeV2, negative_hand-neg,(worst quality:1.4), (low quality:1.4), (monochrome:1.1),text,watermark"

    # 乱数のシードを固定
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # 画像の生成
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt, 
        num_inference_steps=25,
        generator=generator,
        guidance_scale=10,
        # width=512,
        # height=512,
        ).images[0]
    # 画像の保存
    fileName = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    image.save(f"./save/{fileName}", "PNG")
    # 画像の出力
    with open(f"./save/{fileName}", mode="rb") as f:
        binary = f.read()
    return Response(content=binary, media_type="image/png")

if __name__ == "__main__":
    pipe = make_pipline("./models/checkpoints/Counterfeit-V3.0.safetensors")
    pipe = pipe.to("cuda")
    uvicorn.run(app, host="", port=8000)