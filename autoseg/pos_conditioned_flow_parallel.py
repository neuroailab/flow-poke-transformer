import os
import torch
import argparse
import multiprocessing as mp
from pathlib import Path


def get_h5_files(input_dir: str):
    input_path = Path(input_dir)
    h5_files = sorted(input_path.glob('*.h5'))
    return h5_files


def split_files_across_gpus(h5_files, num_gpus):
    chunk_size = len(h5_files) // num_gpus
    chunks = []

    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            end_idx = len(h5_files)
        else:
            end_idx = (i + 1) * chunk_size

        chunks.append((start_idx, end_idx))

    return chunks


def worker_process(gpu_id, start_idx, end_idx, model_state_dict, all_h5_files, args):

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')

    print(f"\n[GPU {gpu_id}] Worker started: files {start_idx}-{end_idx}")

    print(f"[GPU {gpu_id}] Loading model...")
    model = torch.hub.load(".", "fpt_base", source="local")
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    model.requires_grad_(False)
    print(f"[GPU {gpu_id}] Model loaded on GPU {gpu_id}")

    # flow gen
    from autoseg.pos_conditioned_flow import run_flow_generation

    worker_files = all_h5_files[start_idx:end_idx]

    run_flow_generation(
        model=model,
        h5_files=worker_files,
        output_dir=args.output_dir,
        min_prob=args.min_prob,
        num_offsets=args.num_offsets,
        flow_resolution=args.flow_resolution,
        min_mag=args.min_mag,
        max_mag=args.max_mag,
        device='cuda',
        overwrite=args.overwrite,
        gpu_id=gpu_id,
        max_vis=args.max_vis_per_gpu
    )

    print(f"[GPU {gpu_id}] Worker completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parallel flow generation across GPUs')
    parser.add_argument('--input_dir', type=str, default='/ccn2/u/lilianch/external_repos/flow-poke-transformer/autosegp/pmotion12_minprob_samples/h5')
    parser.add_argument('--output_dir', type=str, default='/ccn2/u/lilianch/external_repos/flow-poke-transformer/autosegp/flows_pmotion12_0.005minprob_8dist_final')
    parser.add_argument('--min_prob', type=float, default=0.005)
    parser.add_argument('--num_offsets', type=int, default=15)
    parser.add_argument('--flow_resolution', type=int, default=256)
    parser.add_argument('--min_mag', type=float, default=10.0)
    parser.add_argument('--max_mag', type=float, default=25.0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--max_vis_per_gpu', type=int, default=5) # for num visualizations of all flows
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Start index for file subset (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index for file subset (inclusive)')

    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)

    print(f"Parallel Flow Generation")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"GPUs: {gpu_ids} ({num_gpus} total)")
    print(f"Params: num_offsets={args.num_offsets}, flow_res={args.flow_resolution}")
    print(f"        min_mag={args.min_mag}, max_mag={args.max_mag}, min_prob={args.min_prob}")

    if args.start_idx is not None or args.end_idx is not None:
        print(f"Subset: files {args.start_idx}-{args.end_idx} (inclusive)")

    # load model once, pass in
    print("Loading model on CPU (will be copied to each GPU)")
    model = torch.hub.load(".", "fpt_base", source="local")
    model_state_dict = model.state_dict()
    del model
    print(f"Model state dict loaded\n")

    # split work
    all_h5_files = get_h5_files(args.input_dir)
    total_files = len(all_h5_files)
    print(f"Total H5 files: {total_files}")

    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx + 1 if args.end_idx is not None else total_files  # +1 for inclusive end
        all_h5_files = all_h5_files[start:end]
        print(
            f"Processing subset: files {start} to {args.end_idx if args.end_idx is not None else total_files - 1} ({len(all_h5_files)} files)")
    else:
        print(f"Processing all files: {len(all_h5_files)} files")

    chunks = split_files_across_gpus(all_h5_files, num_gpus)

    for i, (gpu_id, (start_idx, end_idx)) in enumerate(zip(gpu_ids, chunks)):
        print(f"GPU {gpu_id}: files {start_idx}-{end_idx} ({end_idx - start_idx} files)")

    print(f"\nLaunching {num_gpus} workers...\n")

    mp.set_start_method('spawn', force=True)

    processes = []
    for gpu_id, (start_idx, end_idx) in zip(gpu_ids, chunks):
        proc = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                start_idx,
                end_idx,
                model_state_dict,
                all_h5_files,
                args
            )
        )
        proc.start()
        processes.append(proc)

    for i, proc in enumerate(processes):
        proc.join()
        print(f"Worker {i} (GPU {gpu_ids[i]}) finished")

    print("Yay done")
    print(f"Results saved to: {args.output_dir}")