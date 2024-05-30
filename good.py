from deeprobust.graph.data import Dataset
import torch.nn as nn
from utils import *
from model.MLP import MLP
from model.GCN import GCN
import torch.nn.functional as F
import time
import numpy as np
import argparse

'''
This is an example of how to test GOOD-AT on a perturbed graph. We choose cora with 415 perturbations, which is 
generated from PGD in the unit test. If you want to conduct experiments on your own perturbed graphs, you just need to 
modify the dataset load module in this code.
'''

parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--seed', type=int, default=int(time.time()), help='random seed')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')

# Hyperparameters of GOOD-AT
parser.add_argument('--K', type=int, default=20, help='the number of detectors')
parser.add_argument('--d_epochs', type=int, default=50, help='the number of epochs for training the detector')
parser.add_argument('--d_batchsize', type=int, default=2048, help='the batch size for training the detector')
parser.add_argument('--ptb_d', type=float, default=0.3, help='the perturbation rate for training the detector')
parser.add_argument('--threshold', type=float, default=0.1, help='the threshold of detecting adversarial edges')





args = parser.parse_args()

# random seed
if __name__ == '__main__':
    # Set the random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Clearn dataset loading
    data = Dataset(root='c:/tmp/', name=args.dataset, setting='nettack')
    # data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    features = sparse_mx_to_torch_sparse_tensor(features).to_dense()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    n_class = labels.max().item() + 1

    # This perturbed graph is derived from the unit test of PGD against GCN
    ptb_data = np.load('./perturbed_graph/pgd/cora_ml-gcn-415.npz')
    perturbed_adj = ptb_data['perturbed_adj']
    idx_train = ptb_data['idx_train']
    idx_val = ptb_data['idx_val']
    idx_test = ptb_data['idx_test']

    # Train a GCN on the clean graph
    epochs = 300
    ce_loss = nn.CrossEntropyLoss()
    lr = 1e-2
    dropout = 0.9
    weight_decay = 1e-3
    verbose = True
    iters = 200

    model = GCN(features.shape[1], 64, n_class)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    adj_norm = normalize_adj_tensor(adj, sparse=True).to(device)
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)

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
                   adj, labels, pseudo_labels, device, idx_train, idx_val, idx_test, model, batch_size)
        detectors.append(detector)

    print("====== Start evaluating GOOD-AT on perturbed graph")

    perturbed_adj = torch.tensor(perturbed_adj)
    perturbed_adj = perturbed_adj.squeeze(0)
    ptb_adj_norm = normalize_adj_tensor(perturbed_adj, sparse=True).to(device)
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

    change = (perturbed_adj - adj.cpu())
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



