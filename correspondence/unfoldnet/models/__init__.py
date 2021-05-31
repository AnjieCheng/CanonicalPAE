import torch
import torch.nn as nn

from correspondence.unfoldnet.models import decoder

# not supposely to be used
# Decoder dictionary
decoder_dict = {
    'ImplicitFun': decoder.ImplicitFun,
}

class ImplicitNet(nn.Module):
    ''' Unfold Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, encoder=None, fold=None, unfold=None, device=None):
        super().__init__()

        if encoder is not None:
            self.Encoder = encoder.to(device)
        else:
            self.Encoder = None

        if fold is not None:
            self.Fold = fold.to(device)
        else:
            self.Fold = None

        if unfold is not None:
            self.Unfold = unfold.to(device)
        else:
            self.Unfold = None

        self._device = device

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''
        z = self.Encoder(inputs)

        return z


    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model