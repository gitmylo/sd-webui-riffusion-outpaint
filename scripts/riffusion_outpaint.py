from PIL import Image, ImageDraw

from modules import scripts, sd_samplers
import gradio as gr
from modules import img2img, txt2img
import re

from modules.processing import process_images, StableDiffusionProcessing

process_step_pattern = r"(?<!\\)\\{({|\()((?!}|\)).+?)(}|\))}"  # flags: re.DOTALL
comments_pattern = r"\n? *?//.*\n?|\n?/\*[\s\S\n]*?\*/\n?"  # flags: none


class RiffusionOutpaint(scripts.Script):
    def __init__(self):
        scripts.Script.__init__(self)
        self.original_prompt = ("", "")

    def title(self):
        return "Riffusion outpaint"

    def ui(self, is_img2img):
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                enabled = gr.Checkbox(False, label="Enable riffusion outpaint")
                scripts_enabled = gr.Checkbox(False, label="Enable \\{{scripts}}. (UNSAFE)")
            with gr.Row():
                keep_generated = gr.Checkbox(False, label="Keep generated chunks separately")
                keep_debug = gr.Checkbox(False, label="Keep debug images (inpaint base, inpaint mask, full chunk)")
            inpainting_fill_mode = gr.Radio(
                label="masked content",
                choices=["fill", "original", "latent noise", "latent nothing"],
                value="fill",
                type="index",
            )
            inpaint_full_res = gr.Checkbox(True, label="Inpaint full res")
            length = gr.Slider(label="Length", value=2, minimum=2, maximum=128, step=1)
            with gr.Row():
                expand_amount = gr.Slider(label="Expand amount (Higher uses more VRAM)", value=1, minimum=0.1, step=0.1)
                keep_amount = gr.Slider(label="Keep amount AKA \"memory\" (Higher uses more VRAM)", value=1, minimum=0,
                                        maximum=10, step=0.1)
            with gr.Row():
                transition_padding = gr.Slider(label="Overmask (like blur, but only spread, higher is smoother)",
                                               value=64, minimum=0, maximum=512, step=1)
                denoising_strength = gr.Slider(label="Denoising strength", value=1, minimum=0, maximum=1, step=0.01)
            with gr.Accordion("Script info", open=False):
                gr.HTML("Scripts allow you to run python code during the processing of your images.")
                with gr.Accordion("Usage", open=False):
                    gr.HTML("<p>There are 2 types of scripts, \\{(eval)} and \\{{exec}}.<br><b>Eval scripts</b> need "
                            "to be a single statement, you cannot assign variables in them, see examples for "
                            "examples of eval.<br><b>Exec scripts</b> can do just about anything, including not "
                            "returning any value. But for simple expressions, eval is more recommended. see "
                            "examples for examples of exec.</p>")
                with gr.Accordion("Basic info", open=False):
                    gradio_create_html_description("\"info\" object",
                                                   f"The info object contains some information, such as: "
                                                   f"{create_html_list(['<b>info.step</b> - the current step, starts at 0', '<b>info.total_steps</b> - the total amount of steps', '<b>info.p</b> - the processing element'])}")
                    gradio_create_html_description("Comments",
                                                   "You can use // line comments<br>or /* block comments */,"
                                                   " they will be excluded from your prompt")
                with gr.Accordion("Examples", open=False):
                    with gr.Accordion("\\{(eval)}", open=False):
                        gradio_create_html_description("Include step count in prompt", "<b>Generation step: "
                                                       "\\{(info.step+1)}/\\{(info.total_steps)}</b>"
                                                       "<br><b>Note:</b> step is directly accessible too, but it's "
                                                       "suggested you use info.step instead, for more consistency with "
                                                       "exec expressions")
                        gradio_create_html_description("Prompt switching", "<b>\\{([\"lo fi\", \"rap\"][info.step])}</b>"
                                                       "<br><b>Note:</b> this example only works for 2 steps, as the "
                                                       "array with prompts is only 2 in size.<br>This script changes the"
                                                       " section of the prompt from \"lo fi\" to \"rap\"")
                    with gr.Accordion("\\{{exec}}", open=False):
                        gradio_create_html_description("Basic example", "<b>\\{{return \"Testing exec\"}}</b>")
                        gradio_create_html_description("Value assignment", "<b>\\{{info.test = \"test data\"}}</b>")
                    with gr.Accordion("Combined", open=False):
                        gradio_create_html_description("Custom info data", "<b>\\{{info.test = \"test data\"}}\\{(info.test)}</b>")
        return [enabled, scripts_enabled, keep_generated, keep_debug, inpainting_fill_mode, length, expand_amount,
                keep_amount, transition_padding, denoising_strength, inpaint_full_res]

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process(self, p: StableDiffusionProcessing, enabled=False, scripts_enabled=False, keep_generated=False,
                keep_debug=False, inpainting_fill_mode="fill", length=2, expand_amount=1, keep_amount=1,
                transition_padding=64, denoising_strength=1, inpaint_full_res=False):
        if not enabled:
            return p

        self.original_prompt = (p.all_prompts[0], p.all_negative_prompts[0])
        if scripts_enabled:
            process_for_step(0, length, p)

    def postprocess(self, p: StableDiffusionProcessing, processed, enabled=False, scripts_enabled=False,
                    keep_generated=False, keep_debug=False, inpainting_fill_mode="fill", length=2, expand_amount=1,
                    keep_amount=1, transition_padding=64, denoising_strength=1, inpaint_full_res=False):
        if enabled:
            total = processed.images[0]
            for i in range(length - 1):
                if scripts_enabled:  # Only run if scripts are enabled
                    (pos, neg) = self.original_prompt
                    p.prompt = pos
                    p.negative_prompt = neg
                    process_for_step(i + 1, length, p)  # other steps
                (next_chunk, total) = generate_next_chunk(keep_debug, inpainting_fill_mode, length, expand_amount,
                                                          keep_amount, transition_padding, denoising_strength, total,
                                                          inpaint_full_res, p, processed)
            if not keep_generated:
                processed.images.clear()
            processed.images.insert(0, total)


def process_for_step(step, total_steps, p: StableDiffusionProcessing):
    p.all_prompts[0] = p.prompt = replace_prompt_for_step(p.prompt, step, total_steps, p)
    print(f"\nEvaluated Prompt: \"{p.prompt}\"")
    p.all_negative_prompts[0] = p.negative_prompt = replace_prompt_for_step(p.negative_prompt, step, total_steps, p)
    print(f"Evaluated negative prompt: \"{p.negative_prompt}\"")


def replace_prompt_for_step(prompt_text, step, total_steps, p: StableDiffusionProcessing):
    prompt_text = re.sub(comments_pattern, "", prompt_text, 0)

    class InfoObject:
        def __init__(self, step, total_steps, p: StableDiffusionProcessing):
            self.step = step
            self.total_steps = total_steps
            self.p = p

    info = InfoObject(step, total_steps, p)
    while 1:  # Loop until break on the final step
        lastmatches = re.search(process_step_pattern, prompt_text, re.DOTALL)
        if not lastmatches:
            break
        (code, func_call, close_type, replace_start, replace_end) = (lastmatches.group(2), lastmatches.group(1)
                                                                     .replace("(", "eval").replace("{", "exec"),
                                                                     lastmatches.group(3), lastmatches.start(),
                                                                     lastmatches.end())

        if not check_func_call(func_call, close_type):
            continue  # Check if correct closing syntax is used, since I can't do this in regex
        evaluated = ""
        if func_call == "eval":
            evaluated = eval(code)
        elif func_call == "exec":
            # def exec_return_internal():  # This function will be replaced by the executed code
            #     return "This shouldn't happen"
            exec(f"def exec_return_internal(info):\n{code}"
                 .replace("\n", "\n  "),
                 globals())  # Replace \n to \n\t to put everything inside the def, so it's indented

            evaluated = exec_return_internal(info)

        # If you're only assigning, there won't be a return value, which will be falsy
        prompt_text = re.sub(process_step_pattern, "" if evaluated is None else str(evaluated), prompt_text, 1)

    return prompt_text


def check_func_call(func_call, close_type):
    return (func_call == "eval" and close_type == ")") or (func_call == "exec" and close_type == "}")


def generate_next_chunk(keep_debug, inpainting_fill_mode, length, expand_amount, keep_amount, transition_padding,
                        denoising_strength, total, inpaint_full_res, p: StableDiffusionProcessing, processed):
    full_mask_width = int(p.width * (expand_amount + keep_amount))
    keep_amount_width = int(p.width * keep_amount)
    expand_amount_width = int(p.width * expand_amount)
    inpaint_mask = Image.new("RGB", (full_mask_width, p.height), "white")
    inpaint_mask_editing = ImageDraw.Draw(inpaint_mask)
    inpaint_mask_editing.rectangle((keep_amount_width - transition_padding, 0, inpaint_mask.width + transition_padding,
                                    inpaint_mask.height), fill="black")
    inpaint_source = Image.new("RGB", (inpaint_mask.width, inpaint_mask.height), "white")
    for x in range(0, full_mask_width + 1 + total.width, total.width):  # expands as much as needed
        inpaint_source.paste(total, (full_mask_width - x, 0))

    totalout = Image.new("RGB", (total.width + expand_amount_width, total.height), "white")
    totalout.paste(total, (0, 0))
    try:
        generate_with_inpaint_source = generate_img2img(inpaint_source, inpaint_mask, inpainting_fill_mode,
                                                        denoising_strength, transition_padding, inpaint_full_res, p,
                                                        processed)
    except Exception as e:
        print(e)
        generate_with_inpaint_source = inpaint_mask  # black
    generated = generate_with_inpaint_source.crop((generate_with_inpaint_source.width - expand_amount_width -
                                                   (transition_padding * 2), 0, generate_with_inpaint_source.width,
                                                   generate_with_inpaint_source.height))

    if keep_debug:  # for debugging purposes
        processed.images.append(inpaint_mask)
        processed.images.append(inpaint_source)
        processed.images.append(generate_with_inpaint_source)

    totalout.paste(generated, (totalout.width - generated.width, 0))

    processed.images.append(generated)
    return generated, totalout


def generate_img2img(init_img_inpaint, init_mask_inpaint, inpainting_fill, denoising_stength, transition_padding,
                     inpaint_full_res, p: StableDiffusionProcessing, processed):
    sampler_index = 0
    for i in range(len(sd_samplers.samplers_for_img2img)):
        if sd_samplers.samplers_for_img2img[i].name.lower() == p.sampler_name.lower():
            sampler_index = i
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
        transition_padding,  # mask blur: int
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
        init_img_inpaint.height,  # height
        init_img_inpaint.width,  # width
        0,  # resize_mode: int
        inpaint_full_res,  # inpaint_full_res: bool
        0,  # inpaint_full_res_padding: int
        True,  # inpainting_mask_invert
        None,  # img2img_batch_input_dir: str
        None,  # img2img_batch_output_dir: str
        None,  # img2img_batch_inpaint_mask_dir: str
        "",
        # Magic?
        0,  # Script index stuff?
        0,  # Script args?
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


def gradio_create_html_description(title, description):
    gr.HTML(f"<h2>{title}</h2><p>{description}</p>")


def create_html_list(items, unordered=True):
    list_html = "<ul>" if unordered else "<ol>"
    for item in items:
        list_html += f"<li>{item}</li>"
    return list_html + "</ul>" if unordered else "</ol>"
