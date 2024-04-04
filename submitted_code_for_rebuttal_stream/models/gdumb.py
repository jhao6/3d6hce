from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    print('Adding arguments for Gdumb')
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Gdumb(ContinualModel):
    NAME = 'gdumb'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Gdumb, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        # update the buffer data
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:inputs.shape[0]])
        # calculate the loss from the buffer
        buf_inputs, buf_target = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
        buf_outputs = self.net(buf_inputs)

        loss =  self.loss(buf_outputs, buf_target)

        loss.backward()
        self.opt.step()


        return loss.item()
