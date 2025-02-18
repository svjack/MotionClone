import gradio as gr
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from motionclone.models.unet import UNet3DConditionModel
from motionclone.pipelines.pipeline_animation import AnimationPipeline
from motionclone.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available
from motionclone.utils.motionclone_functions import *
import json
from motionclone.utils.xformer_attention import *
import os
import numpy as np
import imageio
import shutil
import subprocess

# 权重下载函数
def download_weights():
    try:
        # 创建模型目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/DreamBooth_LoRA", exist_ok=True)
        os.makedirs("models/Motion_Module", exist_ok=True)
        os.makedirs("models/SparseCtrl", exist_ok=True)

        # 下载 Stable Diffusion 模型
        if not os.path.exists("models/StableDiffusion"):
            subprocess.run(["git", "clone", "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5", "models/StableDiffusion"])

        # 下载 DreamBooth LoRA 模型
        if not os.path.exists("models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors"):
            subprocess.run(["wget", "https://huggingface.co/svjack/Realistic-Vision-V6.0-B1/resolve/main/realisticVisionV60B1_v51VAE.safetensors", "-O", "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors"])

        # 下载 Motion Module 模型
        if not os.path.exists("models/Motion_Module/v3_sd15_mm.ckpt"):
            subprocess.run(["wget", "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt", "-O", "models/Motion_Module/v3_sd15_mm.ckpt"])
        if not os.path.exists("models/Motion_Module/v3_sd15_adapter.ckpt"):
            subprocess.run(["wget", "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt", "-O", "models/Motion_Module/v3_sd15_adapter.ckpt"])

        # 下载 SparseCtrl 模型
        if not os.path.exists("models/SparseCtrl/v3_sd15_sparsectrl_rgb.ckpt"):
            subprocess.run(["wget", "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_rgb.ckpt", "-O", "models/SparseCtrl/v3_sd15_sparsectrl_rgb.ckpt"])
        if not os.path.exists("models/SparseCtrl/v3_sd15_sparsectrl_scribble.ckpt"):
            subprocess.run(["wget", "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_scribble.ckpt", "-O", "models/SparseCtrl/v3_sd15_sparsectrl_scribble.ckpt"])

        print("Weights downloaded successfully.")
    except Exception as e:
        print(f"Error downloading weights: {e}")

# 下载权重
download_weights()

# 加载 model_config
model_config_path = "configs/model_config/model_config.yaml"
model_config = OmegaConf.load(model_config_path)

# 硬编码的配置值
config = {
    "motion_module": "models/Motion_Module/v3_sd15_mm.ckpt",
    "dreambooth_path": "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors",
    "model_config": model_config,
    "W": 512,
    "H": 512,
    "L": 16
}

# 写死 pretrained_model_path
pretrained_model_path = "models/StableDiffusion"

# 模型初始化逻辑
def initialize_models():
    # 设置设备
    adopted_dtype = torch.float16
    device = "cuda"
    set_all_seed(42)

    # 加载模型组件
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder").to(device).to(dtype=adopted_dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(device).to(dtype=adopted_dtype)
    
    # 更新配置
    config["width"] = config.get("W", 512)
    config["height"] = config.get("H", 512)
    config["video_length"] = config.get("L", 16)
    
    # 加载模型配置
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=config["model_config"]["unet_additional_kwargs"]).to(device).to(dtype=adopted_dtype)
    
    # 启用 xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    # 创建 pipeline
    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=None,
        scheduler=DDIMScheduler(**config["model_config"]["noise_scheduler_kwargs"]),
    ).to(device)
    
    # 加载权重
    pipeline = load_weights(
        pipeline,
        motion_module_path=config["motion_module"],
        dreambooth_model_path=config["dreambooth_path"],
    ).to(device)
    pipeline.text_encoder.to(dtype=adopted_dtype)
    
    # 加载自定义函数
    pipeline.scheduler.customized_step = schedule_customized_step.__get__(pipeline.scheduler)
    pipeline.scheduler.customized_set_timesteps = schedule_set_timesteps.__get__(pipeline.scheduler)
    pipeline.unet.forward = unet_customized_forward.__get__(pipeline.unet)
    pipeline.sample_video = sample_video.__get__(pipeline)
    pipeline.single_step_video = single_step_video.__get__(pipeline)
    pipeline.get_temp_attn_prob = get_temp_attn_prob.__get__(pipeline)
    pipeline.add_noise = add_noise.__get__(pipeline)
    pipeline.compute_temp_loss = compute_temp_loss.__get__(pipeline)
    pipeline.obtain_motion_representation = obtain_motion_representation.__get__(pipeline)
    
    # 冻结 UNet 参数
    for param in pipeline.unet.parameters():
        param.requires_grad = False
    pipeline.input_config, pipeline.unet.input_config = config, config
    
    # 准备 UNet 的 attention 和 conv
    pipeline.unet = prep_unet_attention(pipeline.unet, config["motion_guidance_blocks"])
    pipeline.unet = prep_unet_conv(pipeline.unet)
    pipeline.scheduler.customized_set_timesteps(config["inference_steps"], config["guidance_steps"], config["guidance_scale"], device=device, timestep_spacing_type="uneven")
    
    return pipeline

# 初始化模型
pipeline = initialize_models()

def generate_video(uploaded_video, motion_representation_save_dir, generated_videos_save_dir, visible_gpu, default_seed, without_xformers, cfg_scale, negative_prompt, positive_prompt, inference_steps, guidance_scale, guidance_steps, warm_up_steps, cool_up_steps, motion_guidance_weight, motion_guidance_blocks, add_noise_step, new_prompt, seed):
    # 更新配置
    config.update({
        "cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
        "positive_prompt": positive_prompt,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "guidance_steps": guidance_steps,
        "warm_up_steps": warm_up_steps,
        "cool_up_steps": cool_up_steps,
        "motion_guidance_weight": motion_guidance_weight,
        "motion_guidance_blocks": motion_guidance_blocks,
        "add_noise_step": add_noise_step
    })
    
    # 设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpu or str(os.getenv('CUDA_VISIBLE_DEVICES', 0))
    
    # 创建保存目录
    if not os.path.exists(generated_videos_save_dir):
        os.makedirs(generated_videos_save_dir)
    
    # 处理上传的视频
    if uploaded_video is not None:
        # 将上传的视频保存到指定路径
        video_path = os.path.join(generated_videos_save_dir, os.path.basename(uploaded_video))
        shutil.move(uploaded_video, video_path)
        
        # 更新配置
        config["video_path"] = video_path
        config["new_prompt"] = new_prompt + config.get("positive_prompt", "")
        pipeline.input_config, pipeline.unet.input_config = config, config
        
        # 提取运动表示
        seed_motion = seed if seed is not None else default_seed
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed_motion)
        if not os.path.exists(motion_representation_save_dir):
            os.makedirs(motion_representation_save_dir)
        motion_representation_path = os.path.join(motion_representation_save_dir, os.path.splitext(os.path.basename(config["video_path"]))[0] + '.pt')
        pipeline.obtain_motion_representation(generator=generator, motion_representation_path=motion_representation_path)
        
        # 生成视频
        seed = seed_motion
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed)
        pipeline.input_config.seed = seed
        
        videos = pipeline.sample_video(generator=generator)
        videos = rearrange(videos, "b c f h w -> b f h w c")
        save_path = os.path.join(generated_videos_save_dir, os.path.splitext(os.path.basename(config["video_path"]))[0] + "_" + config["new_prompt"].strip().replace(' ', '_') + str(seed_motion) + "_" + str(seed) + '.mp4')
        videos_uint8 = (videos[0] * 255).astype(np.uint8)
        imageio.mimwrite(save_path, videos_uint8, fps=8)
        print(save_path, "is done")

        return save_path
    else:
        return "No video uploaded."

# 使用 Gradio Blocks 构建界面
with gr.Blocks() as demo:
    # 页面标题和描述
    gr.Markdown("# Text-to-Video Generation")
    gr.Markdown("This tool allows you to generate videos from text prompts using a pre-trained model. Upload a video, provide a new prompt, and adjust the settings to create your custom video.")

    # 主要输入区域
    with gr.Row():
        with gr.Column():
            # 视频上传
            uploaded_video = gr.Video(label="Upload Video", source="upload")
            # 新提示词
            new_prompt = gr.Textbox(label="New Prompt", value="A beautiful scene", lines=2)
            # 种子
            seed = gr.Number(label="Seed", value=42)
            # 生成按钮
            generate_button = gr.Button("Generate Video")

        with gr.Column():
            # 输出视频
            output_video = gr.Video(label="Generated Video")

    # 高级设置区域
    with gr.Accordion("Advanced Settings", open=False):
        with gr.Row():
            with gr.Column():
                motion_representation_save_dir = gr.Textbox(label="Motion Representation Save Dir", value="motion_representation/")
                generated_videos_save_dir = gr.Textbox(label="Generated Videos Save Dir", value="generated_videos")
                visible_gpu = gr.Textbox(label="Visible GPU", value="0")
                default_seed = gr.Number(label="Default Seed", value=2025)
                without_xformers = gr.Checkbox(label="Without Xformers", value=False)
            with gr.Column():
                cfg_scale = gr.Number(label="CFG Scale", value=7.5)
                negative_prompt = gr.Textbox(label="Negative Prompt", value="bad anatomy, extra limbs, ugly, deformed, noisy, blurry, distorted, out of focus, poorly drawn face, poorly drawn hands, missing fingers")
                positive_prompt = gr.Textbox(label="Positive Prompt", value="8k, high detailed, best quality, film grain, Fujifilm XT3")
                inference_steps = gr.Number(label="Inference Steps", value=100)
                guidance_scale = gr.Number(label="Guidance Scale", value=0.3)
                guidance_steps = gr.Number(label="Guidance Steps", value=50)
                warm_up_steps = gr.Number(label="Warm Up Steps", value=10)
                cool_up_steps = gr.Number(label="Cool Up Steps", value=10)
                motion_guidance_weight = gr.Number(label="Motion Guidance Weight", value=2000)
                motion_guidance_blocks = gr.Textbox(label="Motion Guidance Blocks", value="['up_blocks.1']")
                add_noise_step = gr.Number(label="Add Noise Step", value=400)

    # 绑定生成函数
    generate_button.click(
        generate_video,
        inputs=[
            uploaded_video, motion_representation_save_dir, generated_videos_save_dir, visible_gpu, default_seed, without_xformers, cfg_scale, negative_prompt, positive_prompt, inference_steps, guidance_scale, guidance_steps, warm_up_steps, cool_up_steps, motion_guidance_weight, motion_guidance_blocks, add_noise_step, new_prompt, seed
        ],
        outputs=output_video
    )

# 启动应用
demo.launch(share = True)
