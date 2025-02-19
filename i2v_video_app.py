import gradio as gr
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from motionclone.models.unet import UNet3DConditionModel
from motionclone.models.sparse_controlnet import SparseControlNetModel
from motionclone.pipelines.pipeline_animation import AnimationPipeline
from motionclone.utils.util import load_weights, auto_download
from diffusers.utils.import_utils import is_xformers_available
from motionclone.utils.motionclone_functions import *
import json
from motionclone.utils.xformer_attention import *
import os
import numpy as np
import imageio
import shutil
import subprocess
from types import SimpleNamespace

# 模型下载逻辑
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

# 模型初始化逻辑
def initialize_models(pretrained_model_path, config):
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
    model_config = OmegaConf.load(config.get("model_config", ""))
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(model_config.unet_additional_kwargs)).to(device).to(dtype=adopted_dtype)
    
    # 加载 controlnet 模型
    controlnet = None
    if config.get("controlnet_path", "") != "":
        assert config.get("controlnet_config", "") != ""
        
        unet.config.num_attention_heads = 8
        unet.config.projection_class_embeddings_input_dim = None

        controlnet_config = OmegaConf.load(config.controlnet_config)
        controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {})).to(device).to(dtype=adopted_dtype)

        auto_download(config.controlnet_path, is_dreambooth_lora=False)
        print(f"loading controlnet checkpoint from {config.controlnet_path} ...")
        controlnet_state_dict = torch.load(config.controlnet_path, map_location="cpu")
        controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
        controlnet_state_dict = {name: param for name, param in controlnet_state_dict.items() if "pos_encoder.pe" not in name}
        controlnet_state_dict.pop("animatediff_config", "")
        controlnet.load_state_dict(controlnet_state_dict)
        del controlnet_state_dict

    # 启用 xformers
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    # 创建 pipeline
    pipeline = AnimationPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler(**model_config.noise_scheduler_kwargs),
    ).to(device)
    
    # 加载权重
    pipeline = load_weights(
        pipeline,
        motion_module_path=config.get("motion_module", ""),
        adapter_lora_path=config.get("adapter_lora_path", ""),
        adapter_lora_scale=config.get("adapter_lora_scale", 1.0),
        dreambooth_model_path=config.get("dreambooth_path", ""),
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
    
    # 冻结 UNet 和 ControlNet 参数
    for param in pipeline.unet.parameters():
        param.requires_grad = False
    if pipeline.controlnet is not None:
        for param in pipeline.controlnet.parameters():
            param.requires_grad = False
    
    pipeline.input_config, pipeline.unet.input_config = SimpleNamespace(**config), SimpleNamespace(**config)
    pipeline.unet = prep_unet_attention(pipeline.unet, config.get("motion_guidance_blocks", []))
    pipeline.unet = prep_unet_conv(pipeline.unet)
    
    return pipeline

# 硬编码的配置值
config = {
    "motion_module": "models/Motion_Module/v3_sd15_mm.ckpt",
    "dreambooth_path": "models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors",
    "model_config": "configs/model_config/model_config.yaml",
    "controlnet_path": "models/SparseCtrl/v3_sd15_sparsectrl_rgb.ckpt",
    "controlnet_config": "configs/sparsectrl/latent_condition.yaml",
    "adapter_lora_path": "models/Motion_Module/v3_sd15_adapter.ckpt",
    "W": 512,
    "H": 512,
    "L": 16,
    "motion_guidance_blocks": ['up_blocks.1'],
}

# 初始化模型
pretrained_model_path = "models/StableDiffusion"
pipeline = initialize_models(pretrained_model_path, config)

# 视频生成函数
def generate_video(uploaded_video, condition_images, new_prompt, seed, motion_representation_save_dir, generated_videos_save_dir, visible_gpu, without_xformers, cfg_scale, negative_prompt, positive_prompt, inference_steps, guidance_scale, guidance_steps, warm_up_steps, cool_up_steps, motion_guidance_weight, motion_guidance_blocks, add_noise_step):
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

    device = pipeline.device
    
    # 创建保存目录
    if not os.path.exists(generated_videos_save_dir):
        os.makedirs(generated_videos_save_dir)
    if not os.path.exists(motion_representation_save_dir):
        os.makedirs(motion_representation_save_dir)
    
    # 处理上传的视频
    if uploaded_video is not None:
        pipeline.scheduler.customized_set_timesteps(config["inference_steps"], config["guidance_steps"], config["guidance_scale"], device=device, timestep_spacing_type="uneven")
        
        # 将上传的视频保存到指定路径
        video_path = os.path.join(generated_videos_save_dir, os.path.basename(uploaded_video))
        shutil.copy2(uploaded_video, video_path)
        
        # 更新配置
        config["video_path"] = video_path
        config["condition_image_path_list"] = condition_images
        config["image_index"] = [0] * len(condition_images)
        config["new_prompt"] = new_prompt + config.get("positive_prompt", "")
        config["controlnet_scale"] = 1.0

        pipeline.input_config, pipeline.unet.input_config = SimpleNamespace(**config), SimpleNamespace(**config)

        # 提取运动表示
        seed_motion = seed if seed is not None else 76739
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed_motion)
        motion_representation_path = os.path.join(motion_representation_save_dir, os.path.splitext(os.path.basename(config["video_path"]))[0] + '.pt')
        pipeline.obtain_motion_representation(generator=generator, motion_representation_path=motion_representation_path, use_controlnet=True)
        
        # 生成视频
        seed = seed_motion
        generator = torch.Generator(device=pipeline.device)
        generator.manual_seed(seed)
        pipeline.input_config.seed = seed
        videos = pipeline.sample_video(generator=generator, add_controlnet=True)

        videos = rearrange(videos, "b c f h w -> b f h w c")
        save_path = os.path.join(generated_videos_save_dir, os.path.splitext(os.path.basename(config["video_path"]))[0] + "_" + config["new_prompt"].strip().replace(' ', '_') + str(seed_motion) + "_" + str(seed) + '.mp4')
        videos_uint8 = (videos[0] * 255).astype(np.uint8)
        imageio.mimwrite(save_path, videos_uint8, fps=8)
        print(save_path, "is done")

        return save_path
    else:
        return "No video uploaded."

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    gr.Markdown("# MotionClone Video Generation")
    with gr.Row():
        with gr.Column():
            uploaded_video = gr.Video(label="Upload Video")
            condition_images = gr.Files(label="Condition Images")
            new_prompt = gr.Textbox(label="New Prompt", value="A beautiful scene")
            seed = gr.Number(label="Seed", value=76739)
            generate_button = gr.Button("Generate Video")
        with gr.Column():
            output_video = gr.Video(label="Generated Video")

    with gr.Accordion("Advanced Settings", open=False):
        motion_representation_save_dir = gr.Textbox(label="Motion Representation Save Dir", value="motion_representation/")
        generated_videos_save_dir = gr.Textbox(label="Generated Videos Save Dir", value="generated_videos/")
        visible_gpu = gr.Textbox(label="Visible GPU", value="0")
        without_xformers = gr.Checkbox(label="Without Xformers", value=False)
        cfg_scale = gr.Number(label="CFG Scale", value=7.5)
        negative_prompt = gr.Textbox(label="Negative Prompt", value="ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers")
        positive_prompt = gr.Textbox(label="Positive Prompt", value="8k, high detailed, best quality, film grain, Fujifilm XT3")
        inference_steps = gr.Number(label="Inference Steps", value=100)
        guidance_scale = gr.Number(label="Guidance Scale", value=0.3)
        guidance_steps = gr.Number(label="Guidance Steps", value=40)
        warm_up_steps = gr.Number(label="Warm Up Steps", value=10)
        cool_up_steps = gr.Number(label="Cool Up Steps", value=10)
        motion_guidance_weight = gr.Number(label="Motion Guidance Weight", value=2000)
        motion_guidance_blocks = gr.Textbox(label="Motion Guidance Blocks", value="['up_blocks.1']")
        add_noise_step = gr.Number(label="Add Noise Step", value=400)

    # 绑定生成函数
    generate_button.click(
        generate_video,
        inputs=[
            uploaded_video, condition_images, new_prompt, seed, motion_representation_save_dir, generated_videos_save_dir, visible_gpu, without_xformers, cfg_scale, negative_prompt, positive_prompt, inference_steps, guidance_scale, guidance_steps, warm_up_steps, cool_up_steps, motion_guidance_weight, motion_guidance_blocks, add_noise_step
        ],
        outputs=output_video
    )

    # 添加示例
    examples = [
        {"video_path": "reference_videos/camera_zoom_out.mp4", "condition_image_paths": ["condition_images/rgb/dog_on_grass.png"], "new_prompt": "Dog, lying on the grass", "seed": 42}
    ]
    examples = list(map(lambda d: [d["video_path"], d["condition_image_paths"], d["new_prompt"], d["seed"]], examples))

    gr.Examples(
        examples=examples,
        inputs=[uploaded_video, condition_images, new_prompt, seed],
        outputs=output_video,
        fn=generate_video,
        cache_examples=False
    )

# 启动应用
demo.launch(share=True)
