import torch
from spikingjelly.activation_based import neuron, functional
import os
import numpy as np
from collections import deque
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import initiate_device
from numpy.lib.recfunctions import structured_to_unstructured
from torchvision.transforms import Resize
import time
import sys
sys.path.append("../")
from model_snn.spiking_model import SpikePoseNet

def process_events(events, height, width):
    tbin = 2
    C, T = 2 *tbin, 8
    sample_size = 100000
    quantization_size = [sample_size//T,1,1]
    w, h = width, height 
    quantized_w = w // quantization_size[1]
    quantized_h = h // quantization_size[2]
    coords = torch.from_numpy(
                    structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.float32))
    coords = torch.floor(coords/torch.tensor(quantization_size)) 
    coords[:, 1].clamp_(min=0, max=quantized_h-1)
    coords[:, 2].clamp_(min=0, max=quantized_w-1)
    tbin_size = quantization_size[0] / tbin
    tbin_coords = (events['t'] % quantization_size[0]) // tbin_size
    tbin_feats = ((events['p']+1) * (tbin_coords+1)) - 1 
    feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*tbin)
    sparse_tensor = torch.sparse_coo_tensor(
        coords.t().to(torch.int32), 
        feats,
        (T, quantized_h, quantized_w, C),
    )
    sparse_tensor = sparse_tensor.coalesce().to_dense()
    return sparse_tensor


def load_model(num_frames=8, model_dir="/media/dulanga/miniconda/pretrained_model/model_snn_8steps.pth"):
    device = torch.device("cuda")
    model_dir=model_dir
    cam_intr = np.array([335.86, 335.86, 128.2,  128.2 ])
    num_frames = num_frames
    model = SpikePoseNet(
        num_frames=num_frames,
        channel=4,
        model_name="sew_resnet34",
        return_interm_layers=True,
        use_tc=False,  # if true, reshape events [B, T, C, H, W] (t=T) to [B, T*C, 1, H, W] (t=T*C)
        spiking_neuron="ParametricLIFNode",
        surrogate_function="ATan",
        cnf="ADD",
        drop_prob=0.1,
        n_layers=2,
        detach_reset=bool(0),
        # hard_reset: reset to 0, soft_reset: substract v_threshold
        v_reset= None,
        cam_intr=cam_intr,
        img_size=256,
        smpl_dir="/media/dulanga/miniconda/smpl_model/smpl/models/SMPL_MALE.pkl",
        batch_size=8,
        use_rnn=0,
        use_recursive=0,
        use_transformer=1,
        n_head=1,
    )

    # set DDP
    model.to(device)
    functional.reset_net(model)
    functional.set_step_mode(model, "m")
    functional.set_backend(model, "cupy", getattr(neuron, "ParametricLIFNode"))
    print("[model_snn dir] model_snn loaded from {}".format(model_dir))
    checkpoint = torch.load(model_dir, map_location="cpu")
    state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    model.load_state_dict(state_dict)
    return model

def test(startime,num_frames=8, model_dir="/media/dulanga/miniconda/pretrained_model/model_snn_8steps.pth"):
    model = load_model(num_frames=num_frames, model_dir=model_dir)
    print('Model load time: ', (time.time()-startime))
    startime = time.time()
    device = torch.device("cuda")
    cam_device = initiate_device("")
    mv_iterator = EventsIterator.from_device(device=cam_device) 
    max_samples = 8
    samples_deque = deque(maxlen=max_samples)
    height, width = mv_iterator.get_size()
    samples = []
    for evs in mv_iterator:
        sparse_tensor =  process_events(evs, height, width)         #T,H,W,C
        samples_deque.append(sparse_tensor)
        evs_count = evs.shape[0]
        if len(samples_deque) == max_samples:
            samples = torch.stack(list(samples_deque))  # B, T, H, W, C
            samples = samples.permute(0, 1, 4, 2, 3)
            samples = Resize((256, 256)).forward(samples.flatten(0, 1)).view(*samples.shape[:3], 256, 256)

            print("------------------------------------- test -----------------------------------------")
            model.eval()
            print('Camera Setup time', (time.time()-startime))
            samples = samples.float().to(device)
            with torch.set_grad_enabled(False):
                startime = time.time()
                for i in range(10):
                    out = model(samples)  # [B, T, C, H, W]
                    functional.reset_net(model)
                    curr_time = time.time()
                    gap=curr_time - startime
                    print('Inference Latency: ', gap)
                    print('Throughput: ', 8*evs.shape[0]/gap)
                    startime = curr_time

def main():
    #seed = 666
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    startime = time.time()
    # set environment
    os.environ["OMP_NUM_THREADS"] = "1"
    # https://discuss.pytorch.org/t/distributed-data-parallel-freezes-without-error-message/8009/27
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_num_threads(1)

    # train
    test(startime)


if __name__ == "__main__":
    main()
