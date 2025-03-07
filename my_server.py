import os
import time
import argparse
from urllib.parse import unquote_plus

from fastapi import FastAPI, Request
import uvicorn

from hy3dgen.text2image import HunyuanDiTPipeline

pipeline_t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device='cuda')
output_folder = os.path.join("../neural-subnet/generate/outputs", "text_to_3d")

app = FastAPI()


@app.post("/text_to_image")
async def text_to_image(request: Request):
    start = time.time()
    params = await request.json()
    # Stage 1: Text to Image
    decoded_prompt = unquote_plus(params['prompt'])
    print(f"get params: {params}， decoded_prompt：{decoded_prompt}")
    seed = params.get('seed', 0)
    steps = params.get('steps', 25)

    image = pipeline_t2i(decoded_prompt, int(seed), int(steps))
    image.save(os.path.join(output_folder, "mesh.png"))

    print(f"Generation time: {time.time() - start}")
    return {"success": True}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9072)
