import argparse
import torch
from torch.utils.data import DataLoader
from MyDataset import MyDataset
from Rotations import Rotations
from model import ConvAttention3D


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="", help="data file name")
    parser.add_argument("--checkpoint", type=str, default="", help="model checkpoint file path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--augment_num", type=int, default=24, help="number of augment")
    parser.add_argument("--feature_num", type=int, default=16, help="number of feature")
    parser.add_argument("--cout_dim", type=int, default=512, help="output dim")
    parser.add_argument("--dimgroup", type=int, default=16, help="group norm dim")
    parser.add_argument("--voxel_size", type=int, default=24, help="feature voxel size")
    parser.add_argument("--num_workers", type=int, default=0, help="number of worker")
    parser.add_argument("--rotation", action="store_true", default=True, help="use rotation(augment)")
    parser.add_argument("--cuda", action="store_true", default=False, help="use cuda")
    args = parser.parse_args()
    return args


def get_rotation_out(args, device, model, data):
    outs = []
    data_temp = data.cpu().numpy()[0]
    for j in range(args.augment_num):
        rot_data = Rotations().rotation(data_temp, j).copy()
        shape = (1, args.feature_num, args.voxel_size, args.voxel_size, args.voxel_size)
        rot_data = torch.from_numpy(rot_data).float().to(device).view(shape)
        outs.append(model(rot_data))
    out = torch.tensor([[torch.mean(torch.tensor(outs))]])
    return out


def inference(args, device, model, test_dataloader):
    model.eval()
    for data in test_dataloader:
        data = data.float().to(device)
        if args.rotation:
            out = get_rotation_out(args, device, model, data)
        else:
            out = model(data)
        print(f"Affinity: {out.item()}")


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    test_dataset = MyDataset(args.filename)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = ConvAttention3D(args).to(device)
    para = torch.load(args.checkpoint)
    model.load_state_dict(para)
    inference(args, device, model, test_dataloader)
