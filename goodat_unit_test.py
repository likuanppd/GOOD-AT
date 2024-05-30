'''
The unit_test module  used in this script is adapted from the following repository:
https://github.com/LoadingByte/are-gnn-defenses-robust

The `perturbed_graph/unit_test.npz` is the direct copy from https://github.com/harishgovardhandamodar/are-gnn-defenses-robust/blob/master/unit_test/unit_test.npz

The gb module used in this script is also obtained from the same repository.

Our usage of the `unit_test` and `gb` modules from the referenced repository ensures comparable and scientific results in our experiments. By utilizing established modules from a reputable source, we maintain consistency with existing methodologies and contribute to the reproducibility of scientific findings in the field.
'''

import json
import torch.nn as nn
from utils import *
from model.MLP import MLP
from model.GCN import GCN
import torch.nn.functional as F
import time
from deeprobust.graph.utils import normalize_adj_tensor
import argparse

parser = argparse.ArgumentParser(description='GoodAt evasion')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
parser.add_argument('--device', type=str, default="cuda:0", help='Number of cuda')
parser.add_argument('--t_in', type=int, default=-9, help='t_in')
parser.add_argument('--t_out', type=int, default=-1, help='t_out')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.9, help='Dropout rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose')
parser.add_argument('--iters', type=int, default=200, help='Number of iterations')
parser.add_argument('--beta', type=float, default=0.001, help='Beta')
parser.add_argument('--run', type=int, default=1, help='Run')
parser.add_argument('--K', type=int, default=20, help='the number of detectors')
parser.add_argument('--ptb_d', type=float, default=0.3, help='the perturbation rate for training the detector')
parser.add_argument('--pgd_epochs', type=int, default=1, help='Number of PGD epochs')
parser.add_argument('--d_epochs', type=int, default=50, help='Number of D epochs')
parser.add_argument('--weight_decay_d', type=float, default=1e-4, help='Weight decay for D')
parser.add_argument('--lr_d', type=float, default=1e-2, help='Learning rate for D')
parser.add_argument('--d_batchsize', type=int, default=2048, help='the batch size for training the detector')
parser.add_argument('--n_hidden_d',type=int,default=64)
parser.add_argument('--sep_datasets',type=list,default=["cora_ml", "citeseer"])
parser.add_argument('--sep_splits',type=list,default=[2])
parser.add_argument('--baseline_model',type=str,default="GoodAt")
parser.add_argument('--scenario_name',type=str,default="evasion")
parser.add_argument('--threshold', type=float, default=0.1, help='the threshold of detecting adversarial edges')


args = parser.parse_args()

device = args.device
epochs = args.epochs
ce_loss = nn.CrossEntropyLoss()
loss_d = nn.BCEWithLogitsLoss()
t_in = args.t_in
t_out = args.t_out
lr = args.lr
dropout = args.dropout
weight_decay = args.weight_decay
verbose = args.verbose
iters = args.iters
beta = args.beta
run = args.run

pgd_epochs = args.pgd_epochs
d_epochs = args.d_epochs
weight_decay_d = args.weight_decay_d
lr_d = args.lr_d
n_hidden_d = args.n_hidden_d
sep_datasets = args.sep_datasets
sep_splits = args.sep_splits
baseline_model = args.baseline_model
scenario_name = args.scenario_name


unit_testfile = "perturbed_graph/unit_test.npz"
with np.load(unit_testfile) as loader:
    loader = dict(loader)

seed = int(time.time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

filename = f'experiment_results_{baseline_model}_{scenario_name}_{sep_datasets}_{sep_splits}.json'

using_deeprobust = False
from utils import train, evaluate
try:
    with open(filename) as f:
        experiment_data_dict = json.load(f)
except:
    experiment_data_dict = {}


for dataset_name in sep_datasets:
    print("dataset_name:", dataset_name)
    A_edges = loader[f"{dataset_name}/dataset/adjacency"]
    X_coords = loader[f"{dataset_name}/dataset/features"]
    y = loader[f"{dataset_name}/dataset/labels"]

    N = y.shape[0]
    D = X_coords[:, 1].max() + 1
    A = np.zeros((N, N))
    A[A_edges[:, 0], A_edges[:, 1]] = 1
    A[A_edges[:, 1], A_edges[:, 0]] = 1
    X = np.zeros((N, D))
    X[X_coords[:, 0], X_coords[:, 1]] = 1

    for split_number in sep_splits: #[0, 1, 2, 3, 4]
        idx_train = loader[f"{dataset_name}/splits/{split_number}/train"]
        idx_val = loader[f"{dataset_name}/splits/{split_number}/val"]
        idx_test = loader[f"{dataset_name}/splits/{split_number}/test"]
        labels = torch.tensor(y).to(device)
        features = torch.tensor(X, dtype=torch.float32).to(device)
        adj = torch.tensor(A, dtype=torch.float32).to(device)
        n_class = labels.max().item() + 1
        model = GCN(features.shape[1], 64, n_class)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        adj_norm = normalize_adj_tensor(adj, sparse=True).to(device)

        for model_name in ["gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gnn", "gnn_guard", "grand", "soft_median_gdc"]:

            prefix = f"{dataset_name}/perturbations/{scenario_name}/{model_name}/split_{split_number}/budget_"

            print("====== Evaluating on the clean graph ======")
            acc = train(model, epochs, optimizer, adj_norm, features, labels, idx_train, idx_val, idx_test, ce_loss,
                        verbose=False)

            # get the pseudo-labels, which will be used to train the detector
            print("====== Get the pseudo-labels ======")
            model.eval()
            pseudo_labels = model(adj_norm, features).argmax(dim=1)

            # Hyper-parameters for detector
            d_epochs = args.d_epochs
            weight_decay_d = 1e-4
            lr_d = 1e-2
            loss_d = nn.BCEWithLogitsLoss()
            batch_size = args.d_batchsize
            n_hidden_d = 64
            dim_input = n_class * 2 + features.shape[1] * 2

            print("====== Start training the detectors ======")
            detectors = []
            for _ in range(args.K):
                detector = MLP(dim_input, 1, n_hidden_d, n_layers=2).to(device)
                optimizer_d = torch.optim.Adam(detector.parameters(), lr=lr_d, weight_decay=weight_decay_d)
                detector = get_detector(detector, optimizer_d, d_epochs, loss_d, args.ptb_d, features,
                                        adj, labels, pseudo_labels, device, idx_train, idx_val, idx_test, model,
                                        batch_size)
                detectors.append(detector)

            # train&test on ptb graph
            for pert_edges in (p for (key, p) in loader.items() if key.startswith(prefix)):
                # The perturbation "pert_edges" as it is stored in the unit test file is just a list of edges that must
                # be flipped. Once again, if you prefer to work with sparse matrices, feel free to do so instead, but
                # remain aware that the list of edges must be symmetrized!
                flipped = 1 - A[pert_edges[:, 0], pert_edges[:, 1]]
                A_perturbed = A.copy()
                A_perturbed[pert_edges[:, 0], pert_edges[:, 1]] = flipped
                A_perturbed[pert_edges[:, 1], pert_edges[:, 0]] = flipped

                print("====== Start evaluating GOOD-AT on perturbed graph")
                ptb_rate = pert_edges.shape[0] / A_edges.shape[0]
                budget = pert_edges.shape[0]
                print("budget:", budget, "， baseline model: ", baseline_model, "on dataset: ", dataset_name, "adapt for: ", model_name)

                labels = torch.tensor(y).to(device)
                features = torch.tensor(X, dtype=torch.float32).to(device)
                perturbed_adj = torch.tensor(A_perturbed, dtype=torch.float32).to(device)
                ptb_adj_norm = normalize_adj_tensor(perturbed_adj, sparse=True).to(device)
                features = features.to(device)
                with torch.no_grad():
                    model.eval()
                    logits_ptb = model(ptb_adj_norm, features)
                logits_ptb = torch.concat((logits_ptb, features), dim=1)
                revise_adj = perturbed_adj.clone()
                threshold = args.threshold
                # remove the adversarial edges by through the detectors
                for edge in torch.triu(perturbed_adj).nonzero():
                    i, j = edge[0].item(), edge[1].item()
                    features_edge = torch.concat((logits_ptb[i], logits_ptb[j]), dim=0)
                    remove_flag = False
                    for k in range(args.K):
                        output = F.sigmoid(detectors[k](features_edge.view(1, -1)))
                        if output > threshold:
                            remove_flag = True
                    if remove_flag:
                        revise_adj[i, j], revise_adj[j, i] = 0, 0
                change = (perturbed_adj - adj)
                removed = perturbed_adj - revise_adj
                cnt = 0
                for edge in torch.triu(removed).nonzero():
                    i, j = edge[0].item(), edge[1].item()
                    if change[i, j] == 1:
                        cnt += 1
                print(cnt)

                revise_adj_norm = normalize_adj_tensor(revise_adj, sparse=True).to(device)
                ptb_acc = evaluate(model, ptb_adj_norm, features, labels, idx_test)
                revise_acc = evaluate(model, revise_adj_norm, features, labels, idx_test)
                print(f'The perturbed accuracy: {ptb_acc}; The revised accuracy: {revise_acc}')


