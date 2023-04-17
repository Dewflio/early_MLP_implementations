## Leonard Puškáč
## FIIT STU, Bratislava
## April 2023

# Master's Thesis -  First Progress Report (DP01) 

In this project, I was assigned to implement and test a method of prevention for a type of attack, the aim of which is to steal the architecture of a neural network. To be specific, this type of attack is called a Side-Channl Attack (SCA) - an attack that utilises the unintended leakage of information through side channels, such as electro-magnetic radiation (EM) or power consumption, to gather relevant information (the values of weights) about a particular neural network architecture running on a microchip. More information about this can be found here [1].

The prevention method we are proposing will rely on randomizing the order of the weights of a layer in the neural network before the actual matrix mulitiplication, so as to disallow the attacker from knowing which weights belong to which neurons. 

In this first part of our project, our aim is to test out the efficacy and the ideal configuration of the proposed prevention method by simulating the attacker's point of view. In other words, we want to test the accuracy of the reconstructed network based on the information that the attacker would be able to gather while our prevention method is applied to the network. Then we want to optimize it for the least overhead while still retaining a low enough accuracy. We are presuming that when we finally implement the actual prevention method the overhead will be increased proportionally to the number of neurons we shuffle. 

[NOTE: WRITE SOME MORE]

# Implementation

In this section I will describe how I implemented the point of view of the attacker - which language and libraries I used, and how exactly the implementation works. The link to my implementation is here:

https://github.com/Dewflio/early_MLP_implementations


## Language

The first decision to make was which language to use. In this case the decision was quite straigh-forward - Python with its simple synatax and abundance of useful libraries is probably the most used language when it comes to implementing any kind of neural network, or when it comes to data science in general. To be specific, I used Python 3.9.16.

## Tensorflow vs Pytorch

The next decision to make was which libraries I wanted to use to make my custom neural network layers. The decision came down to two options - Tensorflow (with Keras) or Pytorch. 

In the end, I decided to use Pytorch for two main reasons:
- Pytoch offers a great deal of customizability when it comes to implementing very specific tweaks to the architecture of the neural network, as opposed to Tensorflow which is a bit more restricted and would require much larger adjustments to achieve what I wanted to do.
- A later stage of this part of the project requires modifying a pre-trained network. Pytorch with libraries like `torchvision.models` allows for a very easy access to pre-trained networks such as resnet. In addition to that, again, the customisability of these pre-trained networks is much easier as opposed to Tensorflow.


## Creating a custom layer

The first thing to figure out was how to make it so that the weights are shuffled before the matrix multiplication occurs on a layer. For that I decided to implement my own layer which inherits from a superclass called `torch.nn.Module` This class, in Pytorch, is used to represent either an entire network (as it can contain other modules and and can perform the forward and backward propagation on them sequentially while keeping all of their relevant parameters, such as weights and biases, in an ordered dictionary called state_dict) or subsections of the network, such as a single layer, which is what we are interested in. 

Here we can see the code used to my first implementation of a custom layer called `ShuffledLinear` which aims to imitate the `nn.Linear` layer (with shuffling):

```
class ShuffleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ShuffleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b = nn.Parameter(torch.Tensor(out_features))
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.W, std=0.1)
        nn.init.constant_(self.b, 0)

    def forward(self, x):
        indices = torch.argsort(torch.rand_like(self.W.T), dim=-1)
        shuffled_W_T = torch.gather(self.W.T, dim=-1, index=indices)

        # Perform the matrix multiplication
        shuffled_output = torch.matmul(x, shuffled_W_T)
        shuffled_output += self.b
        return shuffled_output
```

Later on I implemented a much simpler version of this layer (now called `ShuffledLinear`) which inherits directly from `nn.Linear`:
```
class ShuffledLinear(nn.modules.Linear):
    def __init__(self, *args):
        super(ShuffledLinear, self).__init__(*args)
    def forward(self, input: Tensor) -> Tensor:
        indices = torch.argsort(torch.rand_like(self.weight.T), dim=-1)
        shuffled_W_T = torch.gather(self.weight.T, dim=-1, index=indices)
        shuffled_W = shuffled_W_T.T
        return F.linear(input, shuffled_W, self.bias)
```
And here is the original `forward()` method of `nn.Linear` for reference:
```
def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
```
All this custom layer does is modify the `forward()` method of the `nn.Linear` changing it so that it randomly shuffles the order of weights on each neuron of that layer before calling `F.linear()`.

It does so by first creating a torch tensor with random values and the same shape as the transposed `weight` tensor initialized by the linear layer. This is done using the function `torch.rand_like()`. We do it on a transposed weight tensor because we are interested in shuffling it column-wise, meaning we want each column of the layer to be ordered differently then all the other ones.

Then it creates a tensor of new (shuffled) indices by calling the `torch.argsort()` function, which just assigns an index value to each of the values in the random tensor based on the order of their values sorted.

Next, we create a new weight tensor by calling the function `torch.gather()` using the original transposed weight tensor as the input and the new indices as the template for how the input should be re-ordered. And finally we transpose this new weight tensor to give it its original shape before passing it into the `F.linear()` function.

It is worth mentioning that we can easily do this on other types of layers, not just linear. Here is an example:
```
class ShuffledConv2d(nn.modules.Conv2d):
    def __init__(self, *args):
        super(ShuffledConv2d, self).__init__(*args)
    def forward(self, input: Tensor) -> Tensor:
        indices = torch.argsort(torch.rand_like(self.weight.T), dim=-1)
        shuffled_W_T = torch.gather(self.weight.T, dim=-1, index=indices)
        shuffled_W = shuffled_W_T.T
        return self._conv_forward(input, shuffled_W, self.bias)
```



## Modifying a pre-trained network

In order to try out our custom layer on a fully functioning pre-trained network I had to then figure out how to modify such a network, so that it swaps one of its layers (with pre-defined weight values) for our `ShuffledLinear` layer.

To do this I imported the `resnet18` pre-trained network from `torchvision.models` along with `ResNet18_Weights` values, and tried to figure out how to insert our custom layer into it.

As it turns out, once again, the easiest way to do this was to create a custom `nn.Module` which just incorporates the pretrained `resnet18` into it and swaps one of its children for our `ShuffledLinear` layer. Here is the code:
```
class ShuffledResnet18(nn.Module):
    def __init__(self, weights):
        super(ShuffledResnet18, self).__init__()
        self.resnet_model = resnet18(weights=weights)
        self.layers = []
        for i, child in enumerate(self.resnet_model.children()):
            if i == len(list(self.resnet_model.children())) - 1:
                modified_layer = ShuffledLinear(512, 1000, True)
                modified_layer.weight = child.weight
                modified_layer.bias = child.bias
                self.layers.append(modified_layer)
            else:
                self.layers.append(child)
            pass
        
    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.flatten(x, 1)
            x = layer(x)
        return x

model = ShuffledResnet18(weights=ResNet18_Weights.DEFAULT)
```
What this module does is it initializes a `resnet18` module with the default pre-trained weights and then copies each one of its children (layers) into the `self.layers` array, except for the last one. The last layer of `resnet18` is a linear classification layer with 512 input features, and a 1,000 output features (representing the 1,000 classes that resnet is able to recognize). Instead of copying it we initialize our own `ShuffledLinear` layer to replace it. Then we set the weights and biases of our layer to copy those that are in the original classification layer.

This is reflected in the `forward()` method of our layer. This is the `forward` method of `ResNet` in comparison:
```
def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def forward(self, x):
    return self._forward_impl(x)
```
We can see that the orginal calls all of its children in sequence and then flattens the input before calling the classifier. By copying the all the children we essentially do the same thing in our `forward()` method:
```
def forward(self, x: Tensor) -> Tensor:
    for i, layer in enumerate(self.layers):
        if i == len(self.layers) - 1:
            x = torch.flatten(x, 1)
        x = layer(x)
    return x
```


## Specifying the number of shuffled neurons

Later on in the implementation process it came to my attention that we want to specify the number of neurons we shuffle in order to explore what the resulting accuracy of the network would be if we did not have to shuffled all of them. With it came some modifications to our custom layer:
```
class ShuffledLinearVariable(nn.modules.Linear):
    def __init__(self, *args, n_shuffled=None):
        super(ShuffledLinearVariable, self).__init__(*args)
        self.n_shuffled = n_shuffled if (n_shuffled is not None and n_shuffled < self.in_features) else self.in_features

    def change_n_shuffled(self, n_shuffled: int):
        self.n_shuffled = n_shuffled if (n_shuffled is not None and n_shuffled < self.in_features) else self.in_features
        return self.n_shuffled
    
    def forward(self, input: Tensor) -> Tensor:
        indices = torch.argsort(torch.rand_like(self.weight.T), dim=-1)

        if self.n_shuffled < self.in_features:
            rows_to_sort = random.sample(range(self.in_features), (self.in_features - self.n_shuffled))
            sorted_rows, _ = torch.sort(indices[rows_to_sort], dim=1)
            indices[rows_to_sort] = sorted_rows

        shuffled_W_T = torch.gather(self.weight.T, dim=-1, index=indices)
        shuffled_W = shuffled_W_T.T
        return F.linear(input, shuffled_W, self.bias)
``` 

This solves our need to specify the number of neurons to shuffle. We have a `n_shuffled` attribute which specifies the number of neurons that should be shuffled. It does the same thing as before - creates a tensor of new indices by which we then gather the weight tensor into a new shuffled weight tensor. Except, before gathering, it "unshuffles" some of them. It does so by picking a random sample of rows from indices. The number of elements in the sample is the number of `in_features` i.e. the number of neurons minus the `n_shuffled`. It then unshuffles them by sorting them using `torch.sort()` and saves the shuffled indices into a tensor called `sorted_rows`. Then we just assign `sorted_rows` to `indices` on indices specified in `rows_to_shuffle`.

This also required a modification of the custom resnet module:
```
class ShuffledResnet18Variable(nn.Module):
    def __init__(self, weights):
        super(ShuffledResnet18Variable, self).__init__()
        self.resnet_model = resnet18(weights=weights)
        self.layers = []
        self.modified_layer: ShuffledLinearVariable = None
        for i, child in enumerate(self.resnet_model.children()):
            if i == len(list(self.resnet_model.children())) - 1:
                modified_layer = ShuffledLinearVariable(512, 1000, True)
                modified_layer.weight = child.weight
                modified_layer.bias = child.bias
                self.layers.append(modified_layer)
                self.modified_layer = self.layers[-1]
            else:
                self.layers.append(child)
            pass
    
    def change_n_shuffled(self, n_shuffled: int):
        return self.modified_layer.change_n_shuffled(n_shuffled=n_shuffled)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = torch.flatten(x, 1)
            x = layer(x)
        return x
```

This just gives us the ability to change the number of shuffled neurons during runtime using the method `change_n_shuffled()` if we want to test multiple configurations. 


# Testing Configurations

Finally to test out the overall accuracy of the networks that would be recovered by the attacker I downloaded some validation images that `resnet18` should be able to classify when unaltered. I downloaded them from: https://huggingface.co/datasets/imagenet-1k/tree/main/data - I opted to download the smallest of the files (6.67 GB compressed) called `val_images.tar.gz` that contains 50,000 labelled images which should be enough.

I then tested the accuracy of the network, using different numbers of neurons to shuffle...
[TO BE ADDED - A TABLE OF ACCURACIES]

[ALSO EXPERIMENT WITH SHUFFLING DIFFERENT LAYERS (OTHER THE THE FINAL CLASSIFICATION LAYER)]

# Conclusion

[TO BE ADDED]

# References

[TO BE ADDED]