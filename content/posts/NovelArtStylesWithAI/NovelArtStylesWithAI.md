---
title: "Creating Novel Art Styles With AI"
date: 2023-03-02T01:08:11Z
draft: false
---

The goal of this project is to enable the creation of new, appealing art-styles with AI. The method used to attempt this is training Stable Diffusion LoRa models on specific art styles, then iteratively merging these into new models and comparing their differences mathematically. The training is achieved by preparing a standard set of images, then using ControlNet to 'convert' them into an artist's style.

The project currently consists of a method of training styled LoRa's, and an interactive program that combines and analyses them.

# Groundwork knowledge
## Stable Diffusion

Stable Diffusion is an open-source deep learning model that can generate images from text and other images. A powerful tool called Automatic111 provides a web interface that makes using it much easier.

## LoRa

[LoRa](https://arxiv.org/abs/2106.09685) is a technique for 'fine-tuning' a model such as Stable Diffusion, such that it's output becomes good at outputting certain things, or in our case, outputting in certain style. 

As opposed to techniques like DreamBooth, which fine-tune the entire model, LoRa takes the form of a separate, significantly smaller model that is then applied to SD when images are generated. Using these lets us train, merge and compare significantly faster.

Automatic1111 has in-built support for LoRa's, and even lets multiple be used at once at different weightings.

## ControlNet

[ControlNet](https://arxiv.org/abs/2302.05543) is 'neural network structure' that manages models like Stable Diffusion with additional conditions. In practice, it lets us force SD to output images that replicate the main shapes of other images. 

This means if we want a image to be re-done in a certain style, we pass the original to ControlNet (the 'canny', edge-detection version) with a prompt for the style, and it creates an images with the same shapes and main lines, but re-stylized elements like color and shading. In theory this will aid us in 'isolating' the styles we want by forcing them to be applied in constrained ways.

# Installing the necessary software

Automatic1111 can be found [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Unzip or clone it anywhere you'd like. Be aware you'll need a reasonably good graphics card for it to run.

Assuming you're on Windows, you'll need to edit the ```webui-user``` batch file. Next to ```set PYTHON=``` copy the full path to your Python installation (this should be at version 3.10, if not you can find it (here)[https://www.python.org/ftp/python/3.10.10/python-3.10.10-amd64.exe]). Next to ```set COMMANDLINE_ARGS=``` add ```--xformers``` - this will reduce memory usage.

Running this batch file should bring up a terminal that will give a localhost link to the web application. Once there, go to the Extensions tab, then to Available, click the load button, then search for and install the ControlNet extension. Restart, and it should now be available as a subheading on the image generation screens.

Finally, go [here](https://civitai.com/models/9251/controlnet-pre-trained-models) to download the ControlNet models. Take the 'canny' one, and move it to 'extensions\sd-webui-controlnet\models'.

# LoRa training
## Choosing styles
Any style could be used for this process, but to make things easier we've opted for artists who Stable Diffusion has been trained on and "knows". A list of these can be found archived [here](https://web.archive.org/web/20230311054225/https://github.com/kaikalii/stable-diffusion-artists). Choose styles trained on characters and landscapes, and try to found ones that are distinct to the eye. I opted for Vincent Di Fate, Alan Lee, Michael Garmash, Donato Giancola, Ed Mell and David Mack.

## Standard image set
Styles can be trained on around 20 images or more - the ones I've used are [here](https://drive.google.com/drive/folders/1qjo_yja6L_20WbNeUybiqHB34qveuS7X?usp=sharing), although this set did change between models as I refined it. To try to get a realistic depiction of the style, the images should be varied in composition, but not too noisy/highly detailed or ControlNet will struggle to distinguish lines. They also have to be at 512x512 resolution. [birme.net](birme.net) is useful for cropping in batches.

## Generating stylized images
Go to text2img (img2img also works but influences colours to be close to the original - colour choice is an aspect of style so we leave this to the model itself). Enable ControlNet and set it to canny and the relevant model.

In the prompt box, write "[artist name here] art, [basic description of image subject]". For example, "Alan Lee art, boy walking on beach". Set the sampler to DDIM at 20 steps(there's a lot of conflicting advice on which to choose, this works best for stylized images in my experience. DPM2 a is also a good choice but doesn't produce anything good until around 50 steps).

If the image doesn't follow the shapes of the original, check the second output. This shows the lines ControlNet-canny detected. If they're too messy, increase the low and high threshold - too low, decrease them.

To increase efficiency, you can use the X/Y plot tool in the Scripts section to generate every stylized version of the current image at once. Select Prompt S/R under X type, and writer the names of all the artists you're using in the X values box, comma-separated and starting with one whose name is currently in the prompt. Select "Include Sub Images" and then generate. It'll do all of them at once, and you can find them in "outputs\text2img-images\\[date]".

## Training

Kohya has a useful [Google Colab notebook](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb) for training LoRas. Most of it is self-explanatory if you just go from top-to-bottom. Do note:
- Select StableDiffusion-v1-5 as your model
- After setting the training data directory, open the file manager and add the images to it
- Use the BLIP captioner
- Make sure to put the right path for the pretrained model under Model Config, as well as outputting to Google Drive for convenience
- Set the caption extension to .caption under Dataset Config as this is what BLIP will have created
- Set the epochs to 20 under Training Config, and the save type to .ckpt

Then run Start Training, it'll take the guts of an hour or so.

## Picking the best one

20 .ckpt files will now be in your Drive under "LoRa/output". Move these to "models\Lora" in your Stable Diffusion folder. Pick a suitable prompt, then use the X/Y plot like before, with the X values as the lora names, and the Y value as the strengths (a decimal number between 0 and 1, only a few are needed for comparison's sake). After that's generated you can judge for yourself which model is best - in my experience the 17th usually looks closest to the artist.

# Merging models

Originally I thought you needed the techniques from the next section, flattening, to do this, but it's actually very straightforward. Load the models like so:

```python
import torch
model_a = torch.load("path_to_a.ckpt", map_location="cpu")
model_b = torch.load("path_to_b.ckpt", map_location="cpu")
```

These are both dictionaries full of the model's data. Pick a ratio you want, like 0.5 or 0.3. Then do this:

```python
model_c = dict()
for k in model_a.keys():
    model_c[k] = (ratio * model_a[k]) + ((1 - ratio) * model_b[k])
return model_c
```

PyTorch Tensor objects (the values in the model dictionaries) support arithmetic with normal operators like + and *, which is why this looks so simple.

# Converting a model to a flat array of numbers

The code for doing this can be found at [this GitHub repo](https://github.com/Bemuseed/Blog-Git-Repo). SD models are, in essence, just multi-dimensional arrays of numbers. To mathematically compare them, we have to be able to access them as a single 1D array of these values, 'flattening' their structure.

## How It's Done

Run python from the folder containing the python files. Then:
```python
>>> import flat, tensor_manager
>>> tensors, dict_template = tensor_manager.get_tensors_from_file("my_model.ckpt")
>>> flattened, s, o, l = flat.model_flatten(tensors)
``` 

This should give you a single numpy array of numbers (of length 18874632 for our LoRa models). A corresponding function, ```flat.model_unflatten```, can restore them to their original state given those s, o, and l values, but for our purposes this isn't needed.

## How It Works
### tensor_manager.py

Every .ckpt file is simply a [pickled](https://docs.python.org/3/library/pickle.html) PyTorch model, which can be loaded back into an object using [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load). These objects are just dictionaries, whose values are the model Tensor objects.

The ```get_tensors``` method iterates through those values and puts them in a list, and also sets them to 0 to create a 'template' that can be used to repackage the tensors back into the right format, using ```restore_tensor_dict``` (which simply reverses the process).

### flat.py
First, the ```flatten``` function in flat.py.
```python
def flatten(matrix):
    shapes = [a.shape for a in matrix]
    offsets = [a.size for a in matrix]
    offsets = numpy.cumsum([0] + offsets)
    result = numpy.concatenate([a.flat for a in matrix])
    return result, shapes, offsets
```

1.    The function first creates a list of shapes, where each shape corresponds to the shape of the corresponding numpy array in the input list. This is done using a list comprehension and the "shape" attribute of numpy arrays.
2.    The function then creates a list of offsets, where each offset corresponds to the size (i.e., the number of elements) of the corresponding numpy array in the input list. This is done using a list comprehension and the "size" attribute of numpy arrays.
3.    The offsets list is then converted to a cumulative sum using numpy.cumsum(), giving the starting index of each numpy array in the flattened output.
4.    The function then flattens each numpy array in the input list using the "flat" attribute of numpy arrays and concatenates them together using numpy.concatenate().
5.    Finally, the function returns the flattened output, the list of shapes, and the list of offsets.

The shapes and offsets we get from this are then put to use when we ```unflatten```:
```python
def unflatten(flattened, shapes, offsets):
    restored = numpy.array([numpy.reshape(flattened[offsets[i]:offsets[i + 1]], shape) for i, shape in enumerate(shapes)])
    return restored
```

This part is better understood if we expand it into a longer for-loop:

```python
restored = numpy.array(dtype=numpy.float16)
for i, shape in enumerate(shapes):
	slice_start = offsets[i]
	slice_end = offsets[i + 1]
	slice = flattened[slice_start:slice_end]
	restored_slice = numpy.reshape(slice, shape)
	restored = numpy.append(restored, restored_slice)
```
	

```list_flatten``` simply iterates ```flatten``` over a list of arrays (which is what we are given when we run ```get_tensors_from_file```), saving an additional list called ```limits``` that marks where in the flattened array the individual arrays start and end. These limits are used to mark each ```section``` used in ```list_unflatten```, which also iterates its non-list companion function.

# Comparison

The comparison code in ```model_compare``` looks like this:

```python
def compare(model_a_weights, model_b_weights):
    diff = numpy.abs(model_a_weights - model_b_weights)
    return numpy.average(diff)
```

numpy's functions make this very simple. An array of the difference between every value in the two model's weights is made, then averaged to a single value.

# The interactive application

This is a fairly straightforward command-line tool. The general working is this:

- A 'pool' of models whose data is currently loaded
- Functions accessible from the menu for
    - Loading from disk
    - Saving to disk
    - Deleting models from pool
    - Comparing models in the pool, then ranking and choosing which to keep
    - Merging every unique combination of models in the pool, then comparing, ranking and choosing what to keep
    - Iteratively running the above process

I'll explain the last one more closely. Here's it's corresponding code:

```python
correct = False
    while not correct:
        print()
        iterations = max(1, int(input("Number of iterations: ")))
        ratios = min(9, int(input("Number of merges per model pair (1 means a 0.5 merge, 2 means a 0.33 and 0.66, etc.): ")))
        to_keep = min(1, int(input("Number of models to keep after each iteration: ")))
        print("Iterations:", str(iterations), ", Ratios: ", str(ratios), ", Models to keep: ", str(to_keep))
        inp = input("Is this correct [y/n]: ").lower()
        if inp == "y" or inp == "yes":
            correct = True

    for i in range(iterations):
        print("Iteration " + str(i + 1))
        merged_models, merged_model_names, flattened_merged_models = self.merge_all_and_compare(ratios)
        print("Adding first " + str(to_keep) + " to pool... ", end="")
        self.models.extend(merged_models[:to_keep])
        self.model_names.extend(merged_model_names[:to_keep])
        self.flattened_models.extend(flattened_merged_models[:to_keep])
        print("done.")
    
    print("Iterations complete.")
```
At the top, the user choices for number of iterations, number of ratios (thinking of the possible combinations between two models as a straight line, these are the evenly spaced points along it we are looking at), and the number of models to add to the pool each-time (going down from the most different to least).

Then we loop for the number of iterations, and within it, get the merged models and their data from the function ```merge_all_and_compare```, which is also used by the merging menu item in a much simpler way. These are already in order from most different to least, so we can add their data to the pool (which has three lists for the different data types).

# Next steps for the project

If I had more time, here's what I would do/explore next

- General improvements to application, particularly input checks (such that a mistype can never cause a crash).
- Direct comparison between merged LoRa models and the Dreambooth equivalents (given the same training data etc.).
- Using human evaluation via surveys to confirm if mathematical difference is a good metric for stylistic difference.
