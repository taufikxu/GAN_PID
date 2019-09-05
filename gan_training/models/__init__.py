from gan_training.models import (
    mlp,
    resnet,
    resnet1,
    resnet2,
    resnet3,
    resnet4,
)

generator_dict = {
    'mlp1': mlp.Generator,
    'resnet': resnet.Generator,
    'resnet1': resnet1.Generator,
    'resnet2': resnet2.Generator,
    'resnet3': resnet3.Generator,
    'resnet4': resnet4.Generator,
}

discriminator_dict = {
    'mlp1': mlp.Discriminator,
    'resnet': resnet.Discriminator,
    'resnet1': resnet1.Discriminator,
    'resnet2': resnet2.Discriminator,
    'resnet3': resnet3.Discriminator,
    'resnet4': resnet4.Discriminator,
}
