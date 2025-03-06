"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import random
import sys
import tempfile
import time
import uuid
from io import BytesIO
from urllib.parse import unquote_plus

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Body

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."
output_folder = os.path.join("../neural-subnet/generate/outputs", "text_to_3d")
handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--gen_steps", default=30, type=int)
    parser.add_argument("--octree_resolution", default=256, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--mc_algo", default="mc", type=str)
    parser.add_argument("--max_faces_num", default=90000, type=int)
    parser.add_argument("--t2i_seed", default=0, type=int)
    parser.add_argument("--t2i_steps", default=25, type=int)
    parser.add_argument("--port", default=8084, type=int)
    args = parser.parse_args()
    logger.info(f"args: {args}")
    return args


args = get_args()


class ModelWorker:
    def __init__(self, model_path='tencent/Hunyuan3D-2', device='cuda'):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        self.rembg = BackgroundRemover()
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, device=device)
        self.pipeline_t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
                                               device=device)
        self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(model_path)

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            if 'text' in params:
                text = params["text"]
                image = self.pipeline_t2i(text, args.t2i_seed, args.t2i_steps)
                image.save(os.path.join(output_folder, "mesh.png"))
            else:
                raise ValueError("No input image or text provided")

        params['image'] = image

        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            params['generator'] = torch.Generator(self.device).manual_seed(args.gen_seed)
            params['octree_resolution'] = args.octree_resolution
            params['num_inference_steps'] = args.gen_steps
            params['guidance_scale'] = args.guidance_scale
            params['mc_algo'] = args.mc_algo
            mesh = self.pipeline(**params)[0]

        # if params.get('texture', False):
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=args.max_faces_num)
        mesh = self.pipeline_tex(mesh, image)

        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            temp_file.close()
            os.unlink(temp_file.name)
            # save_path = os.path.join(SAVE_DIR, f'{str(uid)}.glb')
            mesh.export(os.path.join(output_folder, "mesh.glb"))

        torch.cuda.empty_cache()
        return output_folder, uid

app = FastAPI()
@app.post("/generate_from_text")
async def text_to_3d(prompt: str = Body()):
    # Stage 1: Text to Image
    decoded_prompt = unquote_plus(prompt)
    uid = random.randint(1000, 99999)
    print(f"uid:{uid}, prompt: {decoded_prompt}")

    start = time.time()
    params = {"text": decoded_prompt}
    folder, _ = worker.generate(uid, params)

    print(f"Generation time: {time.time() - start}")
    return {"success": True, "path": folder}


@app.post("/generate_from_image")
async def image_to_3d(image_path: str):
    print(f"image_path: {image_path}")

    start = time.time()
    params = {"image": image_path}
    folder, _ = worker.generate(186, params)

    print(f"Generation time: {time.time() - start}")
    return {"message": "3D model generated successfully from image", "output_folder": folder}


if __name__ == "__main__":
    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(model_path=args.model_path, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
