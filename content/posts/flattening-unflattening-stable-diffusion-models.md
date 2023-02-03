---
title: "A Guide to flattening and unflattening Pytorch models"
date: 2023-02-03T10:42:09Z
draft: false
---

## Introduction

A Pytorch model, like any neural net, is essentially a list of floating point numbers in a particular structure. In this guide, you'll be shown how to:
- Extract this structure from a .ckpt file.
- "Flatten" this structure into a 1D Numpy array 
- Reformat this array back to the original format of the model.
- Repackage this into a .ckpt file once more.

## Extraction

A .ckpt file is just a Python dictionary that's been saved to disk using the 'pickle' module. The Pytorch module has [a load method](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load) for loading this into a program. Run the following code in a REPL:
```
>>> import torch
>>> tensor_dict = torch.load("<name-of-model>.ckpt", map_location="cpu")
```

Feel free to look at the dictionary. It consists of a single key-value pair, with the key being the string ```state_dict```, and the value being another dictionary. _That_ dictionary contains 1131 key-value pairs, each value being the Pytorch ```Tensor``` objects that contain the numbers we're interested in.

Now that we know the structure, we can extract those ```Tensor``` objects into a list, like so:

```
>>> tensors = list(tensor_dict['state_dict'].values()) 
```
We're also going to need to keep that dictionary structure for later when we want to convert back to a .ckpt file. To simplify things, and free up some memory, we'll set all the values to 0, and keep the keys the same.
```
>>> new_dict = {'state_dict: dict()} 
>>> for k in list(tensor_dict['state_dict'].keys()): 
... 	new_dict['state_dict'][k] = 0 
```

## Flattening

Now to actually flatten the tensors. First, we'll convert them all to Numpy arrays, which ```Pytorch``` provides an in-built method for.

```
>>> tensors = [t.numpy() for t in tensors]
```

Then, we perform the flattening itself. First we'll make the following function to take a single tensor array and flatten it:

```
def flatten(matrix):
    shapes = [a.shape for a in matrix]
    offsets = [a.size for a in matrix]
    offsets = numpy.cumsum([0] + offsets)
    result = numpy.concatenate([a.flat for a in matrix])
    return result, shapes, offsets
```

While you could type that out in the REPL, I'd recommend you put this in it's own .py file. Then you can use it in a REPL with an import or with 

```
>>> exec(open("<file>.py).read())
```

Then, we use this function in another function that applies this method to a list of different matrices:

```
def list_flatten(matrix_list):
    result = numpy.array([])
    shapes = []
    offsets = []
    limits = [0]

    for a in matrix_list:
        r, s, o = flatten(a)
        result = numpy.append(result, r)
        shapes.append(s)
        offsets.append(o)
        limits.append(len(result))
    return result, shapes, offsets, limits
```
```
>>> flattened, shapes, offsets, limits = list_flatten(tensors)
```
This should give the desired 1D array of numbers, for you to use as you need to.

## Unflattening

This part is the more complex of the two. To begin, a twin for the flattening function:
```
def unflatten(flattened, shapes, offsets):
    restored = numpy.array([numpy.reshape(flattened[offsets[i]:offsets[i + 1]], shape)
            for i, shape in enumerate(shapes)])
    return restored
```

### A Brief Explanation
There's a lot going on in that middle line, so here's a quick breakdown:
- That ```offsets``` variable we made earlier keeps track of starting and ending indices in the flattened 1D array, and the ```shapes``` variable to track their shapes. 
- A list comprehension is used to loop over both of these variables, storing the current ones in the ```i``` and ```shape``` variables within the comprehension.
- In this part:
```
flattened[offsets[i]:offsets[i + 1]]
```
- ... the slice of the flattened array that corresponds to the current original array is determined. ```offsets[i]``` is the starting index, and ```offsets[i+1]``` the ending index.
-```numpy.reshape is used with this and the ```shape``` variable as its arguments, to reshape the slice of the flat array into its original shape.

### Moving On
And now a corresponding function for ```list_flatten```:
```
def list_unflatten(flattened, shapes, offsets, limits):
    unflattened = []
    for i in range(len(limits) - 1):
        section = flattened[limits[i]:limits[i+1]]
        unflattened.append(unflatten(section, shapes[i], offsets[i]))
    return unflattened
```
Which you can then use with the output of the flattening functions:
```
>>> restored_tensor_list = list_unflatten(flattened, shapes, offsets, limits)
``` 
(A word of warning - running this part will eat up your RAM. Make sure to free some up before running it. Linux users without a lot of memory who can increase their swap partition size should do so.)

## Repackaging

And finally, to take our modified tensor list and place it in a new .ckpt file, we first return it to its original dictionary format. First, we convert the arrays back into ```Tensor``` objects:

```
>>> tensors = [torch.from_numpy(nd) for nd in tensors]
```
Then we use that ```dict_template``` dictionary we prepared earlier, and populate it with our new tensor data:

```
>>> t_counter = 0
>>> new_dict = {'state_dict': dict()}
>>> for k in list(dict_template['state_dict'].keys()):
>>> 	new_dict['state_dict'][k] = tensors[t_counter]
>>>     t_counter += 1
```

And finally, we use the Pytorch save method to produce our new .ckpt file.

```
>>> torch.save(new_dict, "my_model".ckpt)
```

