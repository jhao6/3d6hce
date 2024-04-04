from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    print('Adding arguments for Mega')
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    return parser


class Mega(ContinualModel):
    NAME = 'mega'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Mega, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def end_task(self, dataset):
        samples_per_task = self.args.buffer_size // dataset.N_TASKS
        loader = dataset.not_aug_dataloader(samples_per_task)
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)

            buf_loss = self.loss(buf_outputs, buf_labels)
            loss += (buf_loss.item()/loss.item()) * buf_loss

        loss.backward()
        self.opt.step()
        # self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()
