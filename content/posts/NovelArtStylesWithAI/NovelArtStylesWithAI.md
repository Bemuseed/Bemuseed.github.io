---
title: "Creating Novel Art Styles With AI"
date: 2023-03-02T01:08:11Z
draft: false
---

The goal of this project is to enable the creation of new, appealing art-styles with AI. This is to be done by manipulating Stable Diffusion models trained on pre-existing, specific styles. The aim is that we'll be able to move around the 'space' of all such models, using both model merging and Principal Component Analysis.

# Fine-tuning a model with Dreambooth using Google Colab

[Dreambooth](https://arxiv.org/pdf/2208.12242.pdf) allows a Stable Diffusion model to be further trained upon a particular subject (a character or art-style). Many examples can be found on [civit.ai](civit.ai). Training a Dreambooth model requires access to a GPU with sufficient RAM. Rather than doing this with your own hardware, you can offload the task to Google with their Colab service (which is free).

One such notebook is [The Last Ben's](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast-DreamBooth.ipynb#scrollTo=ZnmQYfZilzY6). The notebook is fairly self explanatory, but in brief:

- Run the cells for dependencies and accessing your Google Drive (you'll need ~2.1GB of free space)
- Run the cell for downloading the model
- Create a session, with a suitable name
- Prepare your training images
	- Obtain 20-30 images that are good examples of the style you want to train upon
	- Use [birme.net](birme.net) to crop these to 512x512, in .jpg format, and to name them all with the format ```<style-name> (x).jpg```, where x increments from 1 for each image.
- Run the training cell, which shouldn't take overly long
- Run the testing cell, and click on the link it gives you to access the Automatic1111 interface in your browser
- Test the model by putting in prompts and running generate. You can adjust the sampler used, number of samples, etc. A good guide on getting best results can be found [here](https://old.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/).

The .ckpt file for you new model should be in your Drive under 'Fast-Dreambooth'. If you want to run this again, or a file you obtained and uploaded to your Drive:
- Run the dependency and Drive access cells
- Use the file explorer on the left to copy the path to the .ckpt file you want
- Run the Testing cell and paste the path into the dialogue box that comes up

# Converting a model to a flat array of numbers

The code for doing this can be found at [this GitHub repo](https://github.com/Bemuseed/Blog-Git-Repo). In order to use PCA, we first have to convert each of our models into a 1D array of its weight values. 

## How It's Done

Run python from the folder containing the python files. Then:
```python
>>> import flat, tensor_manager
>>> tensors, dict_template = tensor_manager.get_tensors_from_file("my_model.ckpt")
>>> flattened, s, o, l = flat.list_flatten(tensors)
``` 

Then, to convert a flattened model back:
```python
>>> restored = flat.list_unflatten(flattened, s, o, l)
>>> t_dict = tensor_manager.restore_tensor_dict(restored, dict_template)
>>> import torch
>>> torch.save(t_dict, open("new_model.ckpt", "wb"))
```

## How It Works

Some parts of the code need explained.

### tensor_manager.py

Every .ckpt file is simply a [pickled](https://docs.python.org/3/library/pickle.html) PyTorch model, which can be loaded back into an object using [torch.load](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load). These objects are just dictionaries, with one key, ```state_dict```, which corresponds to another dictionary whose values are the model tensors.

The ```get_tensors``` method iterates through those values and puts them in a list, and also sets them to 0 to create a 'template' that can be used later to repackage the (modified) tensors back into the right format, using ```restore_tensor_dict``` (which simply reverses the process).

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


