# **WARNING**: This extension is **heavily** outdated and **will not work** on current versions of stable-diffusion-webui
If you really want to use it you could:  
A: Submit a pull request which updates the extension  
B: Use a much older version of stable-diffusion-webui which is compatible with this extension

# sd-webui-riffusion-outpaint
## Table of contents
1. [What does it do?](#what-does-it-do)
2. [Requirements](#requirements)
3. [Known bugs](#known-bugs)
4. [Recommended settings](#recommended-settings)
5. [Script prompts](#script-prompts)
6. [How does it work?](#how-does-it-work)

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

## Known bugs
* ~~Does not work with VladMandic's fork, as the img2img method is changed and is missing the seed_enable_extras
parameter. I opened a [pull request](https://github.com/vladmandic/automatic/pull/316) for this, since this missing causes the plugin to not work anymore. for now,
if you want to use VladMandic's fork, but also want to use this extension, use
[This fork](https://github.com/gitmylo/automatic-fix).~~ - merged
* Shows errors in console when running on fixed VladMandic fork. These can be ignored but are also caused by changes
made in the VladMandic fork.

## Recommended settings
* Main settings:
  * Height: 512, Width: 512 (This is the optimal size for Riffusion)
  * Sampling steps: >16 recommended for Euler a (This is the minimum it will generate good results at)
* Script settings:
  * Img2Img masked content: Fill (The image that is being masked is a repeat of the previously generated section)
    * From best to worst (in my testing, results may vary mostly depending on expand amount):
      * `Fill`: fill the extra generated space with colors from the previously generated parts (generally the best)
      * `Latent nothing`: fill the extra generated space with nothing (Can cause inconsistent transitions, slightly better
        than `Latent noise`)
      * `Latent noise`: fill the extra generated space with random noise (Can cause inconsistent transitions)
      * `Original`: fill the extra generated space with what was there originally (The repeated image, can cause double
        sounds on transitions)
  * Length: 2 (this will generate a 10-second clip (as `5 * 2 == 10`) when using 512x512 resolution)
  * Denoising strength: 1 (This will use the full denoising strength)
  * Expand amount: 64 (This will smooth out transitions somewhat, at 0 you are more likely to hear the transition, works
  by spreading the mask out to the previously generated parts of the image, to make them smoother. (This will extend the
  mask further to the left as well))
  * Fast mode (faster generation):
    * Expand amount: 1 (1 means it will inpaint at original width*(keep amount + 1), note: higher values use more VRAM
    and are not always better.)
    * Keep amount (memory): 1 (1 means it will keep the same width as the starter image for outpainting.
    Recommended to be the same as Expand amount)
  * Precision mode (smaller chunks):
    * Expand amount: 0.5 (0.5 means it will inpaint at original width*(keep amount + 0.5), note: higher values use more
    VRAM and are not always better.)
    * Keep amount (memory): 0.5 (0.5 means it will keep the last half of the starting image for outpainting. Recommended
    to be about half of Expand amount)

## Script prompts
Script prompts (`\{(eval)}` and `\{{exec}}`) are explained in the extension UI's info section.

## How does it work?
1. Generate the initial image.
2. Expand the image.
3. Create an outpainting mask on the expanded area (and spread it).
4. Generate the masked area. Now the song has been extended, Also replaces some of the old area to improve the transition.
5. Cut off the newly generated part, and the updated older part.
6. Repeat from step 2 until the specified length (count) has been reached.
7. Combine the generated chunks into one big image.
