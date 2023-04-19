from PIL import Image, ImageDraw

from modules import scripts, sd_samplers
import gradio as gr
from modules import img2img, txt2img
import numpy as np

from modules.processing import process_images, StableDiffusionProcessing


class RiffusionOutpaint(scripts.Script):
    def __init__(self):
        scripts.Script.__init__(self)

    def title(self):
        return "Riffusion outpaint"

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False):
            enabled = gr.Checkbox(False, label="Enable riffusion outpaint")
            keep_generated = gr.Checkbox(False, label="Keep generated chunks separately")
            inpainting_fill_mode = gr.Radio(
                label="Img2Img masked content",
                choices=["fill", "original", "latent noise", "latent nothing"],
                value="original",
                type="index",
            )
            length = gr.Slider(label="Length", value=2, minimum=2, maximum=128, step=1)
            with gr.Row():
                expand_amount = gr.Slider(label="Expand amount (Higher uses more VRAM)", value=1, minimum=0.1, step=0.1)
                keep_amount = gr.Slider(label="Keep amount AKA \"memory\" (Higher uses more VRAM)", value=1, minimum=0,
                                        maximum=10, step=0.1)
            denoising_strength = gr.Slider(label="Denoising strength", value=1, minimum=0, maximum=1, step=0.01)
        return [enabled, keep_generated, inpainting_fill_mode, length, expand_amount, keep_amount, denoising_strength]

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess(self, p: StableDiffusionProcessing, processed, enabled, keep_generated, inpainting_fill_mode,
                    length=2, expand_amount=1, keep_amount=1, denoising_strength=1):
        if enabled:
            total = processed.images[0]
            for i in range(length - 1):
                (next_chunk, total) = generate_next_chunk(inpainting_fill_mode, length, expand_amount, keep_amount,
                                                      denoising_strength, total, p, processed)
            if not keep_generated:
                processed.images.clear()
            processed.images.insert(0, total)
            # processed.images.append(total)


def generate_next_chunk(inpainting_fill_mode, length, expand_amount, keep_amount, denoising_strength, total,
                        p: StableDiffusionProcessing, processed):
    full_mask_width = int(p.width * (expand_amount + keep_amount))
    keep_amount_width = int(p.width * keep_amount)
    expand_amount_width = int(p.width * expand_amount)
    inpaint_mask = Image.new("RGB", (full_mask_width, p.height), "white")
    inpaint_mask_editing = ImageDraw.Draw(inpaint_mask)
    inpaint_mask_editing.rectangle((keep_amount_width, 0, inpaint_mask.width, inpaint_mask.height),
                                   fill="black")
    # processed.images.append(inpaint_mask)  # for debugging purposes
    one_minus_keep_amount = 1 - keep_amount
    inpaint_source = Image.new("RGB", (inpaint_mask.width, inpaint_mask.height), "white")
    for x in range(0, full_mask_width + abs(p.width*2-total.width) + 1, total.width):  # expands as much as needed
        inpaint_source.paste(total, (full_mask_width - x, 0))
    totalout = Image.new("RGB", (total.width + expand_amount_width, total.height), "white")
    totalout.paste(total, (0, 0))
    try:
        generate_with_inpaint_source = generate_img2img(inpaint_source, inpaint_mask, inpainting_fill_mode, denoising_strength, p, processed)
    except Exception as e:
        print(e)
        generate_with_inpaint_source = inpaint_mask  # black
    generated = generate_with_inpaint_source.crop((generate_with_inpaint_source.width - expand_amount_width, 0, generate_with_inpaint_source.width, generate_with_inpaint_source.height))
    totalout.paste(generated, (totalout.width - generated.width, 0))
    processed.images.append(generated)
    return (generated, totalout)


def generate_img2img(init_img_inpaint, init_mask_inpaint, inpainting_fill, denoising_stength,
                     p: StableDiffusionProcessing, processed):
    sampler_index = 0
    for i in range(len(sd_samplers.samplers_for_img2img)):
        if sd_samplers.samplers_for_img2img[i].name.lower() == p.sampler_name.lower():
            sampler_index = i
            print(sd_samplers.samplers_for_img2img[i].name)
    return img2img.img2img(
        "riffusion_outpaint_img2img",
        4,  # mode: int, inpaint upload mask
        p.prompt,
        p.negative_prompt,
        p.styles,
        None,  # init_img: Any
        None,  # sketch: Any
        None,  # init_img_with_mask: Any
        None,  # inpaint_color_sketch: Any
        None,  # inpaint_color_sketch_orig: Any
        init_img_inpaint,  # init_img_inpaint: Any,  -THIS IS THE INPAINTING IMAGE
        init_mask_inpaint,  # init_mask_inpaint: Any,  -THIS IS THE INPAINTING MASK
        p.steps,
        sampler_index,  # Taken from unprompted: https://github.com/ThereforeGames/unprompted/
        0,  # mask blur: int
        1,  # mask alpha: float
        inpainting_fill,  # inpainting fill -THIS IS THE INPAINTING FILL MODE
        p.restore_faces,
        p.tiling,
        p.n_iter,
        1,  # batch size: int
        p.cfg_scale,
        0,  # image_cfg_scale: float
        denoising_stength,  # denoising_strength: float
        p.seed,
        p.subseed,
        p.subseed_strength,
        p.seed_resize_from_h,
        p.seed_resize_from_w,
        False,  # seed enable extras, idk what this does
        p.height,
        p.width,
        0,  # resize_mode: int
        True,  # inpaint_full_res: bool
        0,  # inpaint_full_res_padding: int
        True,  # inpainting_mask_invert
        None,  # img2img_batch_input_dir: str
        None,  # img2img_batch_output_dir: str
        None,  # img2img_batch_inpaint_mask_dir: str
        "",
        # Magic?
        0,
        0,
        0,
        -1
    )[0][0]  # Why?


def generate_txt2img(p: StableDiffusionProcessing):
    return txt2img.txt2img(
        "riffusion_outpaint_img2img",
        p.prompt,
        p.negative_prompt,
        p.styles,
        p.steps,
        0,  # Only euler a for now while i figure out how to get the sampler index
        p.restore_faces,
        p.tiling,
        p.n_iter,
        1,  # batch size
        p.cfg_scale,
        p.seed,
        p.subseed,
        p.subseed_strength,
        p.seed_resize_from_h,
        p.seed_resize_from_w,
        False,  # seed enable extras, idk what this does
        p.height,
        p.width,
        False, None, 0, None, 0, 0, 0,
        "",
        # Magic?
        0,
        0,
        0,
        -1
    )[0][0]  # Why?
