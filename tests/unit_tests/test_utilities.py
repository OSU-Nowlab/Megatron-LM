import argparse
import os
import torch
import mcr_dl
import megatron.core.parallel_state as ps

class Utils:

    world_size = torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])

    @staticmethod
    def initialize_distributed():
        dist = mcr_dl.get_distributed_engine()
        if not dist.is_initialized():
            print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
            torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            mcr_dl.init_processes(dist_engine=args.distributed_engine, dist_backend=args.distributed_backend)

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        dist = mcr_dl.get_distributed_engine()
        dist.barrier()

    @staticmethod
    def initialize_model_parallel(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1, virtual_pipeline_model_parallel_size = None, pipeline_model_parallel_split_rank = None, **kwargs):
        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank, **kwargs)