import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via A-GEM.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class Ogd(ContinualModel):
    NAME = 'ogd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Ogd, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.grad_dims = []
        self.n_tasks = 10
        self.current_task = 0
        self.mem_cnt = 0
        self.samples_per_task = int(self.args.buffer_size // self.n_tasks)
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.base = [[[] for _ in range(self.args.buffer_size)] for _ in range(self.n_tasks)]

    def end_task(self, dataset):
        # resize the memory for each task
        for i in range(self.n_tasks):
            self.base[i] = self.base[i][: self.samples_per_task]
        self.current_task += 1

        loader = dataset.not_aug_dataloader(self.samples_per_task)
        cur_y, cur_x = next(iter(loader))[1:]
        cur_y, cur_x = cur_y.to(self.device), cur_x.to(self.device)
        loss = self.net(cur_x)
        # get the target loss value
        target_loss = loss[:, cur_y.long()].sum()
        # Update ring buffer storing examples from current task
        target_loss.backward()
        current_gradients = [
            p.grad.view(-1) if p.grad is not None
            else torch.zeros(p.numel(), device=self.device)
            for n, p in self.net.named_parameters()]
        cat_gradients = torch.cat(current_gradients)

        if self.current_task == 1:
            self.base[self.current_task - 1][self.mem_cnt] = cat_gradients.clone()
        else:
            new_gradient = self.grad_proj(cat_gradients)
            for g, new_g in zip(cat_gradients, new_gradient):
                g -= new_g
            self.base[self.current_task - 1][self.mem_cnt] = cat_gradients.clone()
        self.mem_cnt += 1
        if self.mem_cnt >= self.samples_per_task:
            self.mem_cnt = 0

    def grad_proj(self, grad):
        # grad = torch.cat(reference_gradients)
        reference_gradients = torch.zeros(grad.size()).cuda()
        sum_grad = torch.zeros(grad.size()).cuda()
        for j in range(self.samples_per_task):
            for i in range(self.n_tasks):
                if self.base[i][j] != []:
                    sum_grad += self.base[i][j]
        reference_gradients += grad.dot(sum_grad)/sum_grad.dot(sum_grad) * sum_grad
        grad -= reference_gradients
        return grad

    def observe(self, inputs, labels, not_aug_inputs):

        self.zero_grad()
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        if self.current_task >= 1:
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=self.device)
                for n, p in self.net.named_parameters()]
            new_gradient = self.grad_proj(torch.cat(current_gradients))
            count_param = 0
            for n, p in self.net.named_parameters():
                p.grad = new_gradient[count_param: count_param + p.numel()].reshape(p.size())
                count_param += p.numel()

        self.opt.step()
        return loss.item()
