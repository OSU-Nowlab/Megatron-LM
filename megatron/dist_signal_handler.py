import signal

import torch
import mcr_dl


def get_world_size():
    dist = mcr_dl.get_distributed_engine()
    if mcr_dl.is_available() and mcr_dl.is_initialized():
        world_size = mcr_dl.get_world_size()
    else:
        world_size = 1
    return world_size


def get_device(local_rank=None):
    dist = mcr_dl.get_distributed_engine()
    backend = dist.get_backend()
    if backend == 'nccl':
        if local_rank is None:
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{local_rank}')
    elif backend == 'gloo':
        device = torch.device('cpu')
    else:
        raise RuntimeError
    return device


def all_gather_item(item, dtype, group=None, async_op=False, local_rank=None):
    dist = mcr_dl.get_distributed_engine()
    if not dist.is_available() or \
       not dist.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    dist.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig

    def signals_received(self):
        all_received = all_gather_item(
            self._signal_received, dtype=torch.int32
        )
        return all_received

    def __enter__(self):
        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self._signal_received = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
