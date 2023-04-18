# sd-webui-riffusion-outpaint
## Table of contents
1. [What does it do?](#what-does-it-do)
2. [Requirements](#requirements)
3. [Recommended settings](#recommended-settings)
4. [How does it work?](#how-does-it-work)

## What does it do?
* sd-webui-riffusion-outpaint is an extension for
[Automatic1111's Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* Using [Riffusion (main project link)](https://github.com/riffusion/riffusion) it lets you generate a song from a
prompt in multiple steps by taking the last 5 seconds, and outpainting the next. This will allow you to generate audio
clips as long as you want, without the quality loss or VRAM usage from doing it in one go!
* Recommended [Riffusion Automatic1111 extension](https://github.com/enlyth/sd-webui-riffusion)

## Requirements
* You have to have the [riffusion model](https://huggingface.co/riffusion/riffusion-model-v1) loaded. (Using it with a
different model will yield unexpected results).
* You could also use a 2 or 4gb riffusion model, by pruning the model, or changing precision. This can easily be done
with [this extension](https://github.com/Akegarasu/sd-webui-model-converter). This will heavily lower your RAM and VRAM
usage, while still retaining good results.

## Recommended settings
* Main settings:
  * Height: 512, Width: 512 (This is the optimal size for Riffusion)
  * Sampling steps: >16 recommended for Euler a (This is the minimum it will generate good results at)
* Script settings:
  * Img2Img masked content: Original (You can play with the other options, but Original works the best)
  * Length: 2 (this will generate a 10 second clip (as `5 * 2 == 10`) when using 512x512 resolution)
  * Fast mode (faster generation):
    * Expand amount: 2 (2 means it will inpaint at original width*2, note: higher values use more VRAM and
    are not always better.)
    * Keep amount (memory): 1 (1 means it will keep the same width as the starter image for outpainting. Setting this
    higher than Expand amount is not allowed. Recommended to be about half of Expand amount)
  * Precision mode (smaller chunks):
    * Expand amount: 1 (1 means it will inpaint at original width, note: higher values use more VRAM and
    are not always better.)
    * Keep amount (memory): 0.5 (0.5 means it will keep the last half of the starting image for outpainting. Setting
    this higher than Expand amount is not allowed. Recommended to be about half of Expand amount)

## How does it work?
1. Generate the initial image.
2. Expand the image.
3. Create an outpainting mask on the expanded area.
4. Generate the masked area. Now the song has been extended by another 5 seconds.
5. Cut off the last 5 seconds generated.
6. If size < end size, expand. else, continue at step 2.
7. Combine the generated chunks into one big file.