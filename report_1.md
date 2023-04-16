## Leonard Puškáč
## FIIT STU, Bratislava
## April 2023

# Master's Thesis -  First Progress Report (DP01) 

In this project, I was assigned to implement and test a method of prevention for a type of attack, the aim of which is to steal the architecture of a neural network. To be specific, this type of attack is called a Side-Channl Attack (SCA) - an attack that utilises the unintended leakage of information through side channels, such as electro-magnetic radiation (EM) or power consumption, to gather relevant information (the values of weights) about a particular neural network architecture running on a microchip. More information about this can be found here [1].

The prevention method we are proposing will rely on randomizing the order of the weights of a layer in the neural network before the actual matrix mulitiplication, so as to disallow the attacker from knowing which weights belong to which neurons. 

In this first part of our project, our aim is to test out the efficacy and the ideal configuration of the proposed prevention method by simulating the attacker's point of view. In other words, we want to test the accuracy of the reconstructed network based on the information that the attacker would be able to gather while our prevention method is applied to the network. Then we want to optimize it for the least overhead while still retaining a low enough accuracy. We are presuming that when we finally implement the actual prevention method the overhead will be increased proportionally to the number of neurons we shuffle. 
WRITE SOME MORE

# Implementation

In this section I describe what I implemented so far.



## Tensorflow vs Pytorch

The first question to answer was which library to use 

Pytorch was easier to implement, as Tensorflow does not allow for easy modifications.

## Creating a custom layer

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

## Modifying a pre-trained network

lorem ipsum

## Specifying the number of shuffled neurons

lorem ipsum

# Testing Configurations

lorem ipsum

# Conclusion

lorem ipsum