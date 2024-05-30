'''

The unit_test module  used in this script is adapted from the following repository:
https://github.com/LoadingByte/are-gnn-defenses-robust

The `perturbed_graph/unit_test.npz` is the direct copy from https://github.com/harishgovardhandamodar/are-gnn-defenses-robust/blob/master/unit_test/unit_test.npz

The gb module used in this script is also obtained from the same repository.

Our usage of the `unit_test` and `gb` modules from the referenced repository ensures comparable and scientific results in our experiments. By utilizing established modules from a reputable source, we maintain consistency with existing methodologies and contribute to the reproducibility of scientific findings in the field.
'''
import argparse
import json
import time
import torch.nn as nn
import numpy as np
import torch
from collections import OrderedDict
from deeprobust.graph.defense import GCNSVD
from gb import metric
from gb.torchext import mul

parser = argparse.ArgumentParser()

#["gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gnn", "gnn_guard", "grand", "soft_median_gdc"]
parser.add_argument('--baseline_model', type=str, default='svd_gcn')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
                    help='whether use symmetric matrix')
parser.add_argument('--sep_run',type=int,default=0)
parser.add_argument('--sep_datasets',type=str,default="cora_ml")
args = parser.parse_args()
sep_datasets = [args.sep_datasets]
args = parser.parse_args()

unit_testfile = "perturbed_graph/unit_test.npz"
with np.load(unit_testfile) as loader:
    loader = dict(loader)
seed = int(time.time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
runs = args.runs
epochs = args.epochs
dropout = args.dropout
weight_decay = args.weight_decay
lr = args.lr
device = args.device
scenario_name =  "evasion"
baseline_model = args.baseline_model

filename = f'experiment_results_baseline_{baseline_model}_{scenario_name}.json'
if args.sep_run != 0:
    filename = f'experiment_results_baseline_{baseline_model}_{scenario_name}_{sep_datasets}.json'
else:
    sep_datasets = ["cora_ml", "citeseer"]
    filename = f'experiment_results_baseline_{baseline_model}_{scenario_name}_{sep_datasets}.json'
using_deeprobust = False
from utils import baseline_evaluate
try:
    with open(filename) as f:
        experiment_data_dict = json.load(f)
except:
    experiment_data_dict = {}


def train_model(y, X, A, train_nodes, val_nodes, test_nodes, baseline_model, runs):
    ptb_rate = 0
    budget = 0
    print("on clean graph")

    from gb.model import GraphSequential, PreprocessA, PreprocessAUsingXMetric, GCN, RGCN, GNNGuard, GRAND, MLP, SoftMedianPropagation, ProGNN
    from gb import preprocess
    labels = torch.tensor(y)
    features = torch.tensor(X, dtype=torch.float32)
    adj = torch.tensor(A, dtype=torch.float32)
    idx_train = train_nodes
    idx_val = val_nodes
    idx_test = test_nodes
    loss = nn.CrossEntropyLoss()
    n_class = labels.max().item() + 1
    using_deeprobust = False
    acc_total = []
    pro_gnn = False
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)

    for run in range(runs):
        if baseline_model.lower() == "gcn":
            model = GCN(n_feat=features.shape[1],
                        hidden_dims=[64], n_class=n_class, dropout=0.9).to(device)
            pass
        elif baseline_model.lower() == "jaccard_gcn":
            thresh = 0.03
            od_dict = OrderedDict(
                jaccard=PreprocessAUsingXMetric(
                    lambda X: metric.pairwise_jaccard(X) > thresh,
                    lambda A, A_mask: mul(A, A_mask)
                ),
                gcn=GCN(n_feat=features.shape[1],
                        hidden_dims=[64], n_class=n_class, dropout=0.9)
            )
            model = GraphSequential(od_dict).to(device)

        elif baseline_model.lower() == "svd_gcn":
            using_deeprobustSVD = True
            model = GCNSVD(nfeat=features.shape[1],
                           nhid=64, nclass=n_class, device=device, dropout=0.9).to(device)
            model.fit(X, A, y, idx_train, idx_val, k=50, verbose=False, weight_decay=weight_decay)

            acc = baseline_evaluate(model, adj, features, labels, idx_test, using_deeprobustSVD, k=50, device=device)
            acc_total.append(acc)
            print(f"{run} acc:{acc}")
            continue
        elif baseline_model.lower() == "rgcn":
            model = RGCN(n_feat=features.shape[1],
                      hidden_dims=[32], n_class=n_class, dropout=0.6).to(device)
        elif baseline_model.lower() == "pro_gnn":
            model = ProGNN(adj, gnn=GCN(n_feat=features.shape[1],
                        hidden_dims=[32], n_class=n_class)).to(device)

        elif baseline_model.lower() == "gnn_guard":
            model = GNNGuard(n_feat=features.shape[1],
                             hidden_dims=[16], n_class=n_class).to(device)
        elif baseline_model.lower() == "grand":  # 待改

            model = GRAND(
                MLP(n_feat=features.shape[1], n_class=n_class, bias=True, hidden_dims=[32])
            ).to(device)


        elif baseline_model.lower() == "soft_median_gdc":
            od_dict = OrderedDict(
                ppr=PreprocessA(
                    lambda A: preprocess.personalized_page_rank(A, teleport_proba=0.25, neighbors=64)),
                gcn=GCN(
                    n_feat=features.shape[1], n_class=n_class, bias=True,
                    propagation=SoftMedianPropagation(temperature=0.5),
                    loops=False, activation="relu", hidden_dims=[64]
                ).to(device)
            )
            model = GraphSequential(od_dict).to(device)


        if baseline_model.lower() == "pro_gnn":
            model.fit([features], labels, torch.tensor(idx_train).to(device), torch.tensor(idx_val).to(device), max_epochs=3000, patience=50, weight_decay=weight_decay)
            acc = baseline_evaluate(model, adj, features, labels, idx_test, is_prognn=True)
        else:
            model.fit((adj, features), labels, torch.tensor(idx_train).to(device), torch.tensor(idx_val).to(device), max_epochs=3000, patience=50, weight_decay=weight_decay)
            acc = baseline_evaluate(model, adj, features, labels, idx_test)
        acc_total.append(acc)

    print('Mean Accuracy:%f' % np.mean(acc_total))
    print('Standard Deviation:%f' % np.std(acc_total, ddof=1))
    result_mean_acc = np.mean(acc_total)
    if scenario_name not in experiment_data_dict:
        experiment_data_dict[scenario_name] = {}
    if dataset_name not in experiment_data_dict[scenario_name]:
        experiment_data_dict[scenario_name][dataset_name] = {}
    model_split_name = f"{model_name}-{split_number}-{ptb_rate}-{budget}"
    if model_split_name not in experiment_data_dict[scenario_name][dataset_name]:
        experiment_data_dict[scenario_name][dataset_name][model_split_name] = {}
    experiment_data_dict[scenario_name][dataset_name][model_split_name]["mean_acc"] = result_mean_acc
    with open(filename, 'w') as f:
        json.dump(experiment_data_dict, f)
    return model


def test_model(y, X, A, model, using_deeprobustSVD=False, device=None):
    labels = torch.tensor(y)
    features = torch.tensor(X, dtype=torch.float32)
    adj = torch.tensor(A, dtype=torch.float32)
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_test = test_nodes



    acc = baseline_evaluate(model, adj, features, labels, idx_test, using_deeprobustSVD, k=50, device=device)
    print('Test Accuracy:%f' % acc)
    if scenario_name not in experiment_data_dict:
        experiment_data_dict[scenario_name] = {}
    if dataset_name not in experiment_data_dict[scenario_name]:
        experiment_data_dict[scenario_name][dataset_name] = {}
    model_split_name = f"{model_name}-{split_number}-{ptb_rate}-{budget}"
    if model_split_name not in experiment_data_dict[scenario_name][dataset_name]:
        experiment_data_dict[scenario_name][dataset_name][model_split_name] = {}
    experiment_data_dict[scenario_name][dataset_name][model_split_name]["mean_acc"] = acc
    with open(filename, 'w') as f:
        json.dump(experiment_data_dict, f)


# The unit test contains perturbations for two datasets: Cora ML and Citeseer. Because there are multiple versions of
# these datasets in circulation, we supply the right ones as part of the unit test.
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

    for split_number in [0, 1, 2, 3, 4]: #[0, 1, 2, 3, 4]
        train_nodes = loader[f"{dataset_name}/splits/{split_number}/train"]
        val_nodes = loader[f"{dataset_name}/splits/{split_number}/val"]
        test_nodes = loader[f"{dataset_name}/splits/{split_number}/test"]

        for model_name in ["gcn", "jaccard_gcn", "svd_gcn", "rgcn", "pro_gnn", "gnn_guard", "grand", "soft_median_gdc"]:

            prefix = f"{dataset_name}/perturbations/{scenario_name}/{model_name}/split_{split_number}/budget_"

            model = train_model(y, X, A, train_nodes, val_nodes, test_nodes, baseline_model=baseline_model, runs=1)
            for pert_edges in (p for (key, p) in loader.items() if key.startswith(prefix)):

                flipped = 1 - A[pert_edges[:, 0], pert_edges[:, 1]]
                A_perturbed = A.copy()
                A_perturbed[pert_edges[:, 0], pert_edges[:, 1]] = flipped
                A_perturbed[pert_edges[:, 1], pert_edges[:, 0]] = flipped

                ptb_rate = pert_edges.shape[0] / A_edges.shape[0]
                budget = pert_edges.shape[0]
                print("budget:", budget, "， baseline model: ", baseline_model, "on dataset: ", dataset_name, "adapt for: ", model_name)

                labels = torch.tensor(y)
                features = torch.tensor(X, dtype=torch.float32)
                perturbed_adj = torch.tensor(A_perturbed, dtype=torch.float32)
                perturbed_adj = perturbed_adj.to(device)
                features = features.to(device)
                labels = labels.to(device)
                idx_test = test_nodes
                if baseline_model.lower() == "pro_gnn":
                    acc = baseline_evaluate(model, perturbed_adj, features, labels, idx_test, is_prognn=True)
                elif baseline_model.lower() == "svd_gcn":
                    acc = baseline_evaluate(model, perturbed_adj, features, labels, idx_test, using_deeprobustSVD=True, k=50, device=device)
                else:
                    acc = baseline_evaluate(model, perturbed_adj, features, labels, idx_test)
                print('Test Accuracy:%f' % acc)
                if scenario_name not in experiment_data_dict:
                    experiment_data_dict[scenario_name] = {}
                if dataset_name not in experiment_data_dict[scenario_name]:
                    experiment_data_dict[scenario_name][dataset_name] = {}
                model_split_name = f"{model_name}-{split_number}-{ptb_rate}-{budget}"
                if model_split_name not in experiment_data_dict[scenario_name][dataset_name]:
                    experiment_data_dict[scenario_name][dataset_name][model_split_name] = {}
                experiment_data_dict[scenario_name][dataset_name][model_split_name]["mean_acc"] = acc
                with open(filename, 'w') as f:
                    json.dump(experiment_data_dict, f)


