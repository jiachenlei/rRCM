# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import accelerate

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=500, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()



def gather(lst, accelerator):
    for i in range(len(lst)):
        if isinstance(lst[i], torch.Tensor):
            lst[i] = lst[i].clone().detach().to(accelerator.device)
        else:
            lst[i] = torch.tensor(lst[i], device=accelerator.device)

        lst[i] = accelerator.gather(lst[i])

    return lst

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    new_state_dict = {}
    for k,v in checkpoint['state_dict'].items():
        new_state_dict[k[len("module."):]] = v
    base_classifier.load_state_dict(new_state_dict)

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)
    base_classifier.to(device)
    # smoothed_classifier = accelerator.prepare(smoothed_classifier)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    g_cpu = torch.Generator()
    g_cpu.manual_seed(1234)
    sampler = torch.utils.data.SubsetRandomSampler(
        indices=[i for i in range(len(dataset)) if i % args.skip == 0], # choose sample at an interval of `skip`
        generator=g_cpu,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=12,
        sampler = sampler,
    )
    print("start certification")
    loader = accelerator.prepare(loader)
    certified_num = 0
    for x, label in loader:

        # only certify every args.skip examples, and stop after args.max examples
        # if i % args.skip != 0:
        #     continue
        # if i == args.max:
        #     break

        # (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.to(device)
        # label = label.to(device)

        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        label, prediction, radius, correct = gather([label, prediction, radius, correct], accelerator)
        if accelerator.is_main_process:
            for i in range(accelerator.num_processes):
                if certified_num < args.max:
                    # When run certification in DDP mode and max_num % number_of_process != 0
                    # this would lead to some images are certified repeatedly, slightly inflating the results
                    # Consider this, we stop recording results when certified_num == max_number
                    print("{}\t{}\t{:.3}\t{}\t{}".format(
                            label[i].item(), prediction[i].item(), radius[i].item(), correct[i].item(), time_elapsed), file=f, flush=True)


                    # record labels of certified images
                    # if y[i].item() not in debug_all_y:
                    #     debug_all_y[y[i].item()] = 0
                    # debug_all_y[y[i].item()] += 1

                certified_num += 1

    f.close()
