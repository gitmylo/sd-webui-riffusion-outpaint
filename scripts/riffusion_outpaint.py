from modules import scripts


class Script(scripts.Script):
    def __init__(self):
        pass

    def title(self):
        return "Riffusion outpaint"

    def ui(self, is_img2img):
        ui_components = []
        return ui_components

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible  # only show on txt2img (for now)
