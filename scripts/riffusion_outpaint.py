from modules import scripts
import gradio as gr
from modules import img2img, txt2img

from modules.processing import process_images, StableDiffusionProcessing


class RiffusionOutpaint(scripts.Script):
    def __init__(self):
        scripts.Script.__init__(self)

    def title(self):
        return "Riffusion outpaint"

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False):
            enabled = gr.Checkbox(False, label="Enable riffusion outpaint")

        return [enabled]

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible  # only show on txt2img (for now)

    # def process(self, p: StableDiffusionProcessing, *args):
        # current_x = 0
        # first_chunk = txt2img.txt2img(
        #     0,
        #     p.prompt,
        #     p.negative_prompt,
        #     p.styles,
        #     p.steps,
        #     p.sampler,
        #     p.restore_faces,
        #     p.tiling,
        #     p.n_iter,
        #     1,  # batch size
        #     p.cfg_scale,
        #     p.seed,
        #     p.subseed,
        #     p.subseed_strength,
        #     p.seed_resize_from_h,
        #     p.seed_resize_from_w,
        #     False,  # seed enable extras, idk what this does
        #     p.height,
        #     p.width,
        #     False, None, 0, None, 0, 0, 0,
        #     {}
        # )[0]

        # img2img.img2img(
        #     0,
        #     p.prompt,
        #     p.negative_prompt,
        #
        # )
        # p.batch_size = 0  # cancel generation
        # return p
    def postprocess(self, p: StableDiffusionProcessing, processed, enabled):
        if enabled:
            processed.images.append(generate_txt2img(p))


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
