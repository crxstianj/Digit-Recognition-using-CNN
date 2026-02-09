import torch

def get_device():
  if torch.cuda.is_available():
      device = torch.device("cuda")
      tensor_c = torch.ones(2, 3, device=device)
      print("CUDA")

  elif torch.backends.mps.is_available():
      device = torch.device("mps")
      tensor_m = torch.ones(2, 3, device=device)
      print("Mps")

  else:
      device = torch.device("cpu")
      print("No GPU available")
