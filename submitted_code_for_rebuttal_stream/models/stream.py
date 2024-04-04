from utils.buffer import Buffer
from torch.nn import functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    print('Adding arguments for STREAM')
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--gamma', type=float, default=1e-1,
                        help='forgetting threshold.')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='the moving-average parameter')
    return parser


class Stream(ContinualModel):
    NAME = 'stream'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Stream, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.gamma = args.gamma
        self.current_loss = 0.0
        self.past_loss = 0.0
        self.past_avg_loss = 0.0
        self.task_id = 0
        self.m = 0.0

    def end_task(self, dataset):

        self.task_id += 1
        # calculate the memory loss of the current model
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.buffer_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buf_loss = self.loss(buf_outputs, buf_labels)
            print('\n')
            print(f'buff_loss: {buf_loss}')
            self.past_loss = buf_loss.detach()

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        # calculate the moving-average forgetting estimator from buffer
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            buf_loss = self.loss(buf_outputs, buf_labels)
            self.m = self.args.beta * self.m + (1 - self.args.beta) * buf_loss.detach()
        # switching gradient update branch
        if not self.buffer.is_empty() and self.m - self.past_loss > self.gamma:
                buf_loss.backward()
        else:
            loss.backward()
        self.opt.step()
        # update the buffer data
        if self.task_id > 0:
            self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:inputs.shape[0]])
        return loss.item()
