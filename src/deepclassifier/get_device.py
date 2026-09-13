import torch


def get_available_device_num_workers():
    """
    Determines the available device (GPU or CPU) and the number of workers.
    Returns:
        tuple: A tuple containing:
            - device (str or int): The available device
            ('cpu' or the GPU device index).
            - num_workers (int): The number of workers to use
            (8 if GPU is available, 0 if CPU).
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        num_workers = 8
        print(f"GPU is available.\n"
              f"CUDA version: {torch.version.cuda}\n"
              f"GPU device name: {torch.cuda.get_device_name(device)}",
              flush=True)
    else:
        device = 'cpu'
        num_workers = 0
        print("GPU is not available. Using CPU by default.", flush=True)

    return device, num_workers
