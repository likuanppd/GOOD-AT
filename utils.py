from copy import deepcopy
from deeprobust.graph.defense import GCN as GCN_deeprobust
from deeprobust.graph.utils import *
from model.pgd import PGDAttack
import torch.utils.data as Data


def evaluate(model, adj, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(adj, features)
        logits = logits[mask]
        test_labels = labels[mask]
        _, indices = logits.max(dim=1)
        correct = torch.sum(indices==test_labels)
        return correct.item() * 1.0 / test_labels.shape[0]

def baseline_evaluate(model, adj, features, labels, mask, using_deeprobustSVD=False, is_prognn=False, k=None, device=None):
    model.eval()
    with torch.no_grad():
        if is_prognn:
            logits = model(features)
        elif using_deeprobustSVD:
            modified_adj = torch.tensor(model.truncatedSVD(adj.cpu(), k=k)).to(device)
            modified_adj_norm = normalize_adj_tensor(modified_adj)
            logits = model(features, modified_adj_norm)
        else:
            logits = model(adj, features)
        logits = logits[mask]
        test_labels = labels[mask]
        _, indices = logits.max(dim=1)
        correct = torch.sum(indices==test_labels)
        return correct.item() * 1.0 / test_labels.shape[0]

def train(model, epochs, optim, adj, features, labels, idx_train, idx_val, idx_test, loss, verbose=True):
    best_loss_val = 9999
    best_acc_val = 0
    weights = deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        logits = model(adj, features)
        l = loss(logits[idx_train], labels[idx_train])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, features, labels, idx_val)
        val_loss = loss(logits[idx_val], labels[idx_val])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    acc = evaluate(model, adj, features, labels, idx_test)
    print("Test Accuracy {:.4f}".format(acc))
    return acc


def train_MLP(model, epochs, optimizer, train_loader, val_loader, test_loader, loss, device, verbose=True):
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            optimizer.zero_grad()
            l = loss(output, y)
            l.backward()
            optimizer.step()
        n_acc = 0
        loss_total = 0
        n = 0
        best_acc_val = 0
        best_loss_val = 0
        model.eval()
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            n += len(y)
            acc = (pred == y).sum().item()
            n_acc += acc
            l = loss(output, y)
            loss_total += l
        acc_total = n_acc / n
        val_loss = loss_total /n
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc_total > best_acc_val:
            best_acc_val = acc_total
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc_total))
    model.load_state_dict(weights)
    model.eval()
    n_acc = 0
    n = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        n += len(y)
        acc = (pred == y).sum().item()
        n_acc += acc
    return  n_acc / n


def train_detector(model, epochs, optimizer, train_loader, loss, device, verbose=True):
    model.train()
    best_acc = 0
    best_loss =9999
    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            optimizer.zero_grad()
            l = loss(output, y.unsqueeze(1).float())
            l.backward()
            optimizer.step()
        n_acc = 0
        loss_total = 0
        n = 0
        best_acc_val = 0
        best_loss_val = 0
    model.eval()
    n_acc = 0
    n = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        pred = (F.sigmoid(output) > 0.5).type(torch.long).squeeze(1)
        n += len(y)
        acc = (pred == y).sum().item()
        n_acc += acc
    return n_acc / n


def train_inductive(model, epochs, optim, adj, run, features, labels, idx_train, idx_val, idx_test, loss, verbose=True):
    # split the graph
    idx_label = np.union1d(idx_train, idx_val)
    adj_train = adj[idx_label][:, idx_label]
    train_features = features[idx_label]
    train_labels = labels[idx_label]
    idx_val = np.arange(len(idx_train),len(idx_label))
    
    best_loss_val = 9999
    best_acc_val = 0
    weights = deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        logits = model(adj_train, train_features)
        l = loss(logits[:len(idx_train)], train_labels[:len(idx_train)])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj_train, train_features, train_labels, idx_val)
        val_loss = loss(logits[idx_val], train_labels[idx_val])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}"
                      .format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    acc = evaluate(model, adj, features, labels, idx_test)
    print("Run {:02d} Test Accuracy {:.4f}".format(run, acc))
    return acc


def get_psu_labels(logits, pseudo_labels, idx_train, idx_test, k=30, append_idx=True):
    # idx_train = np.array([], dtype='int32')
    if append_idx:
        idx_train = idx_train
    else:
        idx_train = np.array([], dtype='int64')
    pred_labels = torch.argmax(logits, dim=1)
    pred_labels_test = pred_labels[idx_test]
    for label in range(pseudo_labels.max().item() + 1):
        idx_label = idx_test[pred_labels_test==label]
        logits_label = logits[idx_label][:, label]
        if len(logits_label) > k:
            _, idx_topk = torch.topk(logits_label, k)
        else:
            idx_topk = np.arange(len(logits_label))
        idx_topk = idx_label[idx_topk]
        pseudo_labels[idx_topk] = label
        idx_train = np.concatenate((idx_train, idx_topk))
    return idx_train, pseudo_labels


def get_detector(detector, optimizer_d, d_epochs, loss_d, ptb_rate_adv, features,
                   adj, labels, pseudo_labels, device, idx_train, idx_val, idx_test, model, batch_size):
    model.eval()
    n_class = labels.max().item() + 1
    dim_input = n_class * 2 + features.shape[1] * 2
    detector = detector
    detector.train()
    # Attack the graph via PGD using pseudo-labels
    perturbations = int(ptb_rate_adv * (adj.sum() // 2))
    target_gcn = GCN_deeprobust(nfeat=features.shape[1],
          nhid=64,
          nclass=n_class,
          dropout=0.9, device=device, lr=0.01, weight_decay=1e-3)
    target_gcn = target_gcn.to(device)
    target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    attack_model = PGDAttack(model=target_gcn, nnodes=adj.shape[0], loss_type='Tanh', device=device)
    attack_model = attack_model.to(device)
    features_norm = normalize_feature(features.to('cpu'))
    attack_model.attack(features_norm, adj.to('cpu'), pseudo_labels.to('cpu'), idx_test, perturbations, epochs=100)
    adversarial_adj = attack_model.modified_adj
    adversarial_adj = adversarial_adj.to(device)
    change_adj = adversarial_adj - adj
    insert_adj = F.relu(change_adj).int()

    # Construct positive and negtive samples. Positive samples are original edges, and negtive ones are adversarial edges
    adversarial_adj_norm = normalize_adj_tensor(adversarial_adj).to(device)
    with torch.no_grad():
        logits = model(adversarial_adj_norm, features)
    logits = torch.concat((logits, features), dim=1)
    num_edge_adj = int(adj.sum().item() // 2)
    positive_edges = torch.zeros((num_edge_adj, dim_input)).to(device)
    for idx, edge in enumerate(torch.triu(adj).nonzero()):
        i, j =edge[0].item(), edge[1].item()
        positive_edges[idx] = torch.concat((logits[i], logits[j]), dim=0)
    positive_labels = torch.zeros(num_edge_adj).long()

    num_edge_insert = int(insert_adj.sum().item() // 2)
    negtive_edges = torch.zeros((num_edge_insert, dim_input)).to(device)
    for idx, edge in enumerate(torch.triu(insert_adj).nonzero()):
        i, j =edge[0].item(), edge[1].item()
        negtive_edges[idx] = torch.concat((logits[i], logits[j]), dim=0)
    negtive_labels = torch.ones(num_edge_insert).long()

    # Train the detector
    train_features_d = torch.concat((positive_edges, negtive_edges), dim=0)
    train_labels_d = torch.concat((positive_labels, negtive_labels), dim=0)
    train_dataset = Data.TensorDataset(train_features_d, train_labels_d)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    acc = train_detector(detector, d_epochs, optimizer_d, train_loader, loss_d, device, verbose=True)

    return detector
