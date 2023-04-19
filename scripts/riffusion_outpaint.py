from PIL import Image, ImageDraw

from modules import scripts
import gradio as gr
from modules import img2img, txt2img
from torchvision.transforms.functional import to_pil_image, pil_to_tensor

from modules.processing import process_images, StableDiffusionProcessing


class RiffusionOutpaint(scripts.Script):
    def __init__(self):
        scripts.Script.__init__(self)

    def title(self):
        return "Riffusion outpaint"

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False):
            enabled = gr.Checkbox(False, label="Enable riffusion outpaint")
            inpainting_fill_mode = gr.Radio(
                label="Img2Img masked content",
                choices=["fill", "original", "latent noise", "latent nothing"],
                value="original",
                type="index",
            )
            length = gr.Slider(label="Length", value=2, minimum=2, maximum=128, step=1)
            with gr.Row():
                expand_amount = gr.Slider(label="Expand amount (Higher = more VRAM)", value=1, minimum=0.1, step=0.1)
                keep_amount = gr.Slider(label="Keep amount AKA \"memory\" (Higher = more VRAM)", value=1, minimum=0,
                                        maximum=1, step=0.1)  # TODO: expand maximum when it gets implemented
        return [enabled, inpainting_fill_mode, length, expand_amount, keep_amount]

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess(self, p: StableDiffusionProcessing, processed, enabled, inpainting_fill_mode, length, expand_amount, keep_amount):
        if enabled:
            inpaint_mask = Image.new("RGB", (p.width * (expand_amount + keep_amount), p.height), "white")
            inpaint_mask_editing = ImageDraw.Draw(inpaint_mask)
            inpaint_mask_editing.rectangle((p.width * keep_amount, 0, inpaint_mask.width, inpaint_mask.height),
                                           fill="black")
            processed.images.append(inpaint_mask)  # for debugging purposes
            one_minus_keep_amount = 1 - keep_amount
            inpaint_source = Image.new("RGB", (inpaint_mask.width, inpaint_mask.height), "white")
            x = 0
            while x < inpaint_source.width:
                inpaint_source.paste(processed.images[0], (one_minus_keep_amount * inpaint_source.width, 0))
                x -= keep_amount
            processed.images.append(inpaint_source)
            processed.images.append(generate_img2img(inpaint_source, inpaint_mask, inpainting_fill_mode, p, processed))
            # processed.images.append(generate_txt2img(p))


def generate_img2img(init_img_inpaint, init_mask_inpaint, inpainting_fill, p: StableDiffusionProcessing, processed):
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
        0,  # Only euler a for now while i figure out how to get the sampler index
        4,  # mask blur: int
        1,  # mask alpha: float
        inpainting_fill,  # inpainting fill -THIS IS THE INPAINTING FILL MODE
        p.restore_faces,
        p.tiling,
        p.n_iter,
        1,  # batch size: int
        p.cfg_scale,
        0,  # image_cfg_scale: float
        1,  # denoising_strength: float
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
