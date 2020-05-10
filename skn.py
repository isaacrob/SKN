from sklearn.base import BaseEstimator
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import progressbar
import copy

from models import *
from losses import f_loss, reg_betainc
from plots import plot_2d

def change_cluster_labels_to_sequential(clusters):
    labels = np.unique(clusters)
    clusters_to_labels = {cluster:i for i, cluster in enumerate(labels)}
    seq_clusters = np.array([clusters_to_labels[cluster] for cluster in clusters])

    return seq_clusters

def make_cost_matrix(c1, c2):
    c1 = change_cluster_labels_to_sequential(c1)
    c2 = change_cluster_labels_to_sequential(c2)
    
    uc1 = np.unique(c1)
    uc2 = np.unique(c2)
    l1 = uc1.size
    l2 = uc2.size
    assert(l1 == l2 and np.all(uc1 == uc2)), str(uc1) + " vs " + str(uc2)

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size

    return m

def get_accuracy(clusters, labels):
    cost = make_cost_matrix(clusters, labels)
    row_ind, col_ind = linear_sum_assignment(cost)
    to_labels = {i: ind for i, ind in enumerate(col_ind)}
    clusters_as_labels = list(map(to_labels.get, clusters))
    acc = np.sum(clusters_as_labels == labels) / labels.shape[0]

    return acc

def tokens_to_tfidf(x):
    list_of_strs = [' '.join(str(token) for token in item if token != 0) for item in x]
    out = TfidfVectorizer().fit_transform(list_of_strs)

    return out

class SKN(BaseEstimator):
    def __init__(
        self,
        n_components = 2,
        model = 'auto',
        min_neighbors = 1,
        max_neighbors = 20,
        snn = True,
        batch_size = 256,
        ignore = .9,
        metric = 'euclidean',
        neighbors_preprocess = None,
        use_gpu = True,
        learning_rate = 1e-3,
        optimizer_override = None,
        epochs = 10,
        verbose_level = 1,
        random_seed = 37,
        gamma = 1,
        semisupervised = False,
        cluster_subnet_dropout_p = .3,
        is_tokens = False,
        cluster_subsample_n = 1000,
        zero_cutoff = 1e-2,
        internal_dim = 64,
        cluster_subnet_training_epochs = 50,
        semisupervised_weight = None,
        l2_penalty = 1e-3,
        prune_graph = False
    ):
        self.n_components = n_components
        self.model = model
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.snn = snn
        self.batch_size = batch_size
        self.ignore = ignore
        self.metric = metric
        self.neighbors_preprocess = neighbors_preprocess
        self.use_gpu = use_gpu
        self.learning_rate = learning_rate
        self.optimizer_override = optimizer_override
        self.epochs = epochs
        self.verbose_level = verbose_level
        self.random_seed = random_seed
        self.gamma = gamma
        self.semisupervised = semisupervised
        self.cluster_subnet_dropout_p = cluster_subnet_dropout_p
        self.is_tokens = False # forces TF-IDF preprocessing if preprocessing unspecified
        self.cluster_subsample_n = cluster_subsample_n
        self.zero_cutoff = zero_cutoff
        self.internal_dim = internal_dim
        self.cluster_subnet_training_epochs = cluster_subnet_training_epochs
        self.semisupervised_weight = semisupervised_weight
        self.l2_penalty = l2_penalty
        self.prune_graph = prune_graph

    def _print_with_verbosity(self, message, level):
        if level <= self.verbose_level:
            print(message)

    def _progressbar_with_verbosity(self, data, level, max_value = None):
        if level <= self.verbose_level:
            for datum in progressbar.progressbar(data, max_value = max_value):
                yield datum
        else:
            for datum in data:
                yield datum

    def _select_model(self, X):
        # not sure if allowed to modify model attribute under sklearn rules
        n_dims = len(X.shape)
        if type(X) is tuple:
            self._print_with_verbosity("assuming token-based data, using bag-of-words model", 1)
            self.is_tokens = True
            vocab = set()
            for x in X:
                vocab.update(x)
            vocab_size = len(vocab)
            self.model = BOWNN(self.n_components, vocab_size, internal_dim = self.internal_dim)
        elif n_dims == 2:
            self._print_with_verbosity("using fully connected neural network", 1)
            self.model = FFNN(self.n_components, X.shape[1])
        elif n_dims == 4:
            self._print_with_verbosity("using convolutional neural network", 1)
            n_layers = int(np.log2(min(X.shape[2], X.shape[3])))
            self.model = CNN(self.n_components, n_layers, internal_dim = self.internal_dim)
        else:
            assert False, "not sure which neural network to use based off data provided"

    def _get_near_and_far_pairs_mem_efficient_chunks(self, X, block_size = 512, return_sorted = True):
        n_neighbors = self.max_neighbors

        closest = []
        furthest = []
        if type(X) is np.ndarray:
            splits = np.array_split(X, max(X.shape[0] // block_size, 1))
            max_value = len(splits)
        else:
            inds = list(range(0, X.shape[0], block_size))
            inds.append(None)
            splits = (X[inds[i]:inds[i+1]] for i in range(len(inds) - 1))
            max_value = len(inds) - 1

        self._print_with_verbosity(f"using metric {self.metric} to build nearest neighbors graph", 2)

        for first in self._progressbar_with_verbosity(splits, 2, max_value = max_value):
            dists = pairwise_distances(first, X, n_jobs = -1, metric = self.metric)
            # dists = cdist(first, X, metric = metric)
            this_closest = np.argpartition(dists, n_neighbors + 1)[:, :n_neighbors+1]
            if return_sorted:
                original_set = set(this_closest[-1])
                relevant = dists[np.arange(this_closest.shape[0])[:, None], this_closest]
                sorted_inds = np.argsort(relevant)
                this_closest = this_closest[np.arange(sorted_inds.shape[0])[:, None], sorted_inds]
                assert set(this_closest[-1]) == original_set, "something went wrong with sorting"
                this_closest = this_closest[:, 1:]
            closest.append(this_closest)

            probs = dists / np.sum(dists, axis = 1)[:, None]
            this_furthest = np.array([np.random.choice(len(probs[i]), n_neighbors, False, probs[i]) for i in range(len(probs))])
            furthest.append(np.array(this_furthest))

        closest = np.concatenate(closest)
        furthest = np.concatenate(furthest)

        return closest, furthest

    def _build_dataset(self, X, y = None):
        # returns Dataset object 
        neighbors_X = X.view(X.shape[0], -1).cpu().numpy()

        if self.is_tokens and self.neighbors_preprocess is None:
            self._print_with_verbosity("using tokenized data without neighbors preprocessing so using TF-IDF transform", 2)
            self.neighbors_preprocess = tokens_to_tfidf

        if self.neighbors_preprocess is not None:
            neighbors_X = self.neighbors_preprocess(neighbors_X)

        closest, furthest = self._get_near_and_far_pairs_mem_efficient_chunks(neighbors_X)

        samples = []
        paired = []

        # for semisupervised version
        # assuming y has positive integer class labels
        # and -1 if there is no label
        first_label = []
        second_label = []
        
        already_paired = set()
        for first, seconds in enumerate(closest):
            represented = 0
            for ind, second in enumerate(seconds[::-1]): # matters if sorted and min_neighbors so closest are last
                if self.snn:
                    if first not in closest[second]:
                        n_left = len(seconds) - ind
                        if n_left > self.min_neighbors - represented:
                            continue

                if self.semisupervised and self.prune_graph:
                    if y[first] != y[second] and y[first] != -1 and y[second] != -1:
                        continue

                represented += 1

                if tuple(sorted([first, second])) not in already_paired and first != second:
                    first_data = X[first]
                    second_data = X[second]
                    stack = torch.stack([first_data, second_data])

                    samples.append(stack)
                    paired.append(1)
                    already_paired.add(tuple(sorted([first, second])))

                    if y is not None:
                        first_label.append(y[first])
                        second_label.append(y[second])
                    else:
                        first_label.append(-1)
                        second_label.append(-1)

        already_paired = set()
        for first, seconds in enumerate(furthest):
            for second in seconds:
                if self.semisupervised and self.prune_graph:
                    if y[first] == y[second] and y[first] != -1 and y[second] != -1:
                        continue

                if tuple(sorted([first, second])) not in already_paired and first != second:
                    first_data = X[first]
                    second_data = X[second]
                    stack = torch.stack([first_data, second_data])

                    samples.append(stack)
                    paired.append(0)
                    already_paired.add(tuple(sorted([first, second])))

                    if y is not None:
                        first_label.append(y[first])
                        second_label.append(y[second])
                    else:
                        first_label.append(-1)
                        second_label.append(-1)

        samples = torch.stack(samples)
        paired = torch.Tensor(np.array(paired)) 
        first_label = torch.Tensor(np.array(first_label))
        second_label = torch.Tensor(np.array(second_label))

        dataset = TensorDataset(samples, paired, first_label.long(), second_label.long())

        return dataset

    def _train_siamese_one_epoch(self, data_loader):
        epoch_loss = 0
        self.model.train()
        for data, target, first_label, second_label in self._progressbar_with_verbosity(data_loader, 1):
            self.optimizer.zero_grad()

            data = data.to(self.device)
            target = target.to(self.device)
            first_label = first_label.to(self.device)
            second_label = second_label.to(self.device)

            output_1 = self.model(data[:, 0])
            output_2 = self.model(data[:, 1])

            loss = f_loss(output_1, output_2, target, ignore = self.ignore, device = self.device)

            if self.semisupervised:
                which = first_label != -1
                first_label_pred = self.semisupervised_model(data[which, 0])
                loss = loss + F.cross_entropy(first_label_pred, first_label[which])*self.semisupervised_weight
                which = second_label != -1
                second_label_pred = self.semisupervised_model(data[which, 1])
                loss = loss + F.cross_entropy(second_label_pred, second_label[which])*self.semisupervised_weight

            if self.l2_penalty > 0:
                loss = loss + self.l2_penalty*torch.mean((torch.norm(output_1, p = 2, dim = 1) + torch.norm(output_2, p = 2, dim = 1)))

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            del output_1
            del output_2

        self._print_with_verbosity(f"training loss: {epoch_loss / len(data_loader)}", 1)

    def _train_one_epoch(self, model, data_loader, optimizer, crit):
        model.train()
        for data, target in data_loader:
            optimizer.zero_grad()

            data = data.to(self.device)
            target = target.to(self.device)

            pred = model(data)

            loss = crit(pred, target)

            loss.backward()
            optimizer.step()

    def transform(self, X, to_numpy = True, batch_size = 4096, model = None):
        if model is None:
            model = self.model
        # embeds the data
        dataset = TensorDataset(X)
        embed_loader = DataLoader(dataset, shuffle = False, batch_size = batch_size)

        embeddings = []
        model.eval()
        with torch.no_grad():
            for data in embed_loader:
                data = data[0].to(self.device)
                embedding = model(data).cpu()
                if to_numpy:
                    embedding = embedding.numpy()
                embeddings.append(embedding)

        if to_numpy:
            embeddings = np.concatenate(embeddings)
            embeddings = embeddings.reshape(len(X), -1)
        else:
            embeddings = torch.cat(embeddings)
            embeddings = embeddings.view(len(X), -1)
    
        return embeddings    

    def _get_exp_dist(self, data_loader):
        # sets self.exp_dist based off mean of means dist between positive pairs
        self.model.eval()

        cumulative_dist = 0
        with torch.no_grad():
            for data, target, first_label, second_label in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                should_be_close = target == 1
                if torch.sum(should_be_close) == 0:
                    continue

                output_1 = self.model(data[should_be_close, 0])
                output_2 = self.model(data[should_be_close, 1])

                d = torch.norm(output_1 - output_2, p = 2, dim = 1)

                # get parameters for f distribution. not sure these are right..
                d1 = torch.Tensor([output_1.shape[-1]]).to(self.device)
                d2 = torch.Tensor([1]).to(self.device)

                # compute p-value
                p = reg_betainc(d1*d/(d1*d+d2), d1/2, d2/2)
                # reject null hypothesis
                d = d[p < self.ignore]
                # do means
                cumulative_dist += torch.mean(d).item()

                del output_1
                del output_2
                del should_be_close

        avg_dist = cumulative_dist / len(data_loader)

        self.exp_dist = avg_dist

    def _cluster(self, X):
        # runs spectral clustering based off self.exp_dist as Gaussian kernel bandwidth
        # sets self.n_clusters and returns cluster_assignments
        X = X.reshape(X.shape[0], -1)

        n = min(X.shape[0], self.cluster_subsample_n)

        inds = np.random.choice(X.shape[0], n, replace = False)
        D = pairwise_distances(X[inds], n_jobs = -1, metric = 'euclidean')

        sigma = (self.exp_dist*self.gamma)**2
        A = np.exp(-D**2 / sigma)

        sums = A.sum(axis = 1)
        D = np.diag(sums)
        L = D - A

        vals, vecs = eigh(L, eigvals = [0, int(X.shape[0]**.5)]) # assuming only sqrt possible clusters
        # print(vals)

        n_zeros = np.sum(vals < self.zero_cutoff)
        self._print_with_verbosity(f"found {n_zeros} candidate clusters", 3)
        init_clusters = KMeans(n_zeros, n_init = 100).fit_predict(vecs[:, :n_zeros])
        # print(np.unique(init_clusters))

        n_neighbors = int(2*np.log2(X.shape[0]))
        clusters = KNeighborsClassifier(n_neighbors).fit(X[inds], init_clusters).predict(X)
        # print(np.unique(clusters))
        clusters = change_cluster_labels_to_sequential(clusters)
        # print(clusters.shape)
        # print(clusters[:10])

        self.n_clusters = np.unique(clusters).shape[0]

        return clusters

    def predict(self, X, model = None):
        if model is None:
            assert self.best_full_net is not None, "have not trained a prediction network yet!"
            model = self.best_full_net

        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, batch_size = 4096, shuffle = False)
        preds = []

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            for data in progressbar.progressbar(data_loader):
                data = data[0].to(self.device)
                _, pred = torch.max(model(data), 1)
                preds.extend(pred.cpu().numpy())

        preds = np.array(preds)

        return preds

    def _build_cluster_subnet(self, X, transformed, clusters):
        # creates clustering subnet and updates best model
        # sets self.cluster_subnet

        self._print_with_verbosity("training cluster subnet to predict spectral labels", 2)

        cluster_counts = torch.zeros(self.n_clusters).float().to(self.device)
        for cluster_assignment in clusters:
            cluster_counts[cluster_assignment] += 1
        cluster_weights = len(clusters)/self.n_clusters/cluster_counts

        dataset = TensorDataset(torch.Tensor(transformed), torch.Tensor(clusters).long())
        data_loader = DataLoader(dataset, shuffle = True, batch_size = self.batch_size)

        cluster_subnet = ClusterNet(transformed.shape[-1], self.n_clusters)
        cluster_subnet_optimizer = optim.Adam(cluster_subnet.parameters())
        cluster_subnet_crit = nn.CrossEntropyLoss(weight = cluster_weights)
        cluster_subnet.train()
        cluster_subnet = cluster_subnet.to(self.device)

        for i in self._progressbar_with_verbosity(range(self.cluster_subnet_training_epochs), 2):
            self._train_one_epoch(cluster_subnet, data_loader, cluster_subnet_optimizer, cluster_subnet_crit)

        # now fine-tune the whole pipeline
        dataset = TensorDataset(X, torch.Tensor(clusters).long())
        data_loader = DataLoader(dataset, shuffle = True, batch_size = self.batch_size)

        full_net = FullNet(copy.deepcopy(self.model), cluster_subnet)
        full_net_optimizer = optim.Adam(full_net.parameters(), lr = 1e-4)
        full_net_crit = cluster_subnet_crit
        full_net.train()
        full_net = full_net.to(self.device)

        for i in self._progressbar_with_verbosity(range(self.cluster_subnet_training_epochs), 2): # NOTE: in original, this was 20, not 50, by default
            self._train_one_epoch(full_net, data_loader, full_net_optimizer, full_net_crit)

        preds = self.predict(X, model = full_net)

        # new_transformed = self.transform(X, model = full_net.embed_net)

        # delta_mi = silhouette_score(new_transformed, preds)
        delta_mi = mutual_info_score(preds, clusters)
        if delta_mi > self.best_delta_mi:
            self._print_with_verbosity(f"found new best delta mi of {delta_mi}", 1)
            self.best_delta_mi = delta_mi
            self.best_full_net = full_net
            self.best_n_clusters = self.n_clusters

        return preds

    def fit(self, X, y = None, y_for_verification = None, plot = False):
        # assert not self.semisupervised, "semisupervised not supported yet"

        self.best_delta_mi = -1
        self.best_full_net = None
        self.best_n_clusters = 1

        if type(X) is not torch.Tensor:
            X = torch.Tensor(X)

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        use_y_to_verify_performance = y_for_verification is not None
        self.semisupervised = self.semisupervised and y is not None

        if self.semisupervised and self.semisupervised_weight is None:
            self.semisupervised_weight = np.sum(y != -1) / y.shape[0]

        if self.semisupervised:
            n_classes = np.unique(y[y != -1]).shape[0] # because of the -1 

        if use_y_to_verify_performance:
            verify_n_classes = np.unique(y_for_verification).shape[0]
            self._print_with_verbosity(f"number of classes in verification set: {verify_n_classes}", 3)

        if self.model == "auto":
            self._select_model(X)

        self.device = torch.device("cuda") if (torch.cuda.is_available() and self.use_gpu) else torch.device("cpu")

        self._print_with_verbosity(f"using torch device {self.device}", 2)

        self._print_with_verbosity("building dataset", 1)

        dataset = self._build_dataset(
            X, 
            y = y if self.semisupervised else None, 
        )

        data_loader = DataLoader(dataset, shuffle = True, batch_size = self.batch_size)

        self.model = self.model.to(self.device)

        if self.optimizer_override is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        else:
            self.optimizer = self.optimizer_override(self.model.parameters(), lr = self.learning_rate)

        if self.semisupervised:
            label_subnet = ClusterNet(self.n_components, n_classes).to(self.device)
            self.semisupervised_model = FullNet(self.model, label_subnet).to(self.device)
            self.optimizer = optim.Adam(self.semisupervised_model.parameters(), lr = self.learning_rate)

        self._print_with_verbosity("training", 1)

        for i in range(self.epochs):
            self.model.train()
            self._print_with_verbosity(f"this is epoch {i}", 1)
            self._train_siamese_one_epoch(data_loader)
            self.model.eval()
            transformed = self.transform(X)

            self._get_exp_dist(data_loader)
            self._print_with_verbosity(f"found expected distance between related points as {self.exp_dist}", 3)
            cluster_assignments = self._cluster(transformed)
            self._print_with_verbosity(f"found {self.n_clusters} clusters", 1)

            preds = self._build_cluster_subnet(X, transformed, cluster_assignments)

            if use_y_to_verify_performance:
                nmi_score = normalized_mutual_info_score(cluster_assignments, y_for_verification, 'geometric')
                self._print_with_verbosity(f"NMI of cluster labels with y: {nmi_score}", 2)

                nmi_score = normalized_mutual_info_score(preds, y_for_verification, 'geometric')
                self._print_with_verbosity(f"NMI of network predictions with y: {nmi_score}", 1)

                if self.n_clusters == verify_n_classes:
                    acc_score = get_accuracy(cluster_assignments, y_for_verification)
                    self._print_with_verbosity(f"accuracy of cluster labels: {acc_score}", 2)

                if np.unique(preds).shape[0] == verify_n_classes:
                    acc_score = get_accuracy(preds, y_for_verification)
                    self._print_with_verbosity(f"accuracy of network predictions: {acc_score}", 1)
                else:
                    self._print_with_verbosity(f"number of predicted classes did not match number of clusters so not computing accuracy, correct {verify_n_classes} vs {self.n_clusters}", 2)

            if plot and self.n_components == 2:
                plot_2d(transformed, cluster_assignments)

                if use_y_to_verify_performance:
                    plot_2d(transformed, y_for_verification)



if __name__ == "__main__":
    from torchvision.datasets import MNIST, USPS, FashionMNIST, CIFAR10
    from torchtext.datasets import AG_NEWS

    n = 500
    semisupervised_proportion = .2

    e = SKN(n_components = 8, internal_dim = 16, epochs = 30, semisupervised = True, verbose_level = 3, gamma = .5)

    USPS_data_train = USPS("./", train = True, download = True)
    USPS_data_test = USPS("./", train = False, download = True)
    USPS_data = ConcatDataset([USPS_data_test, USPS_data_train])
    X, y = zip(*USPS_data)

    y_numpy = np.array(y[:n])
    X_numpy = np.array([np.asarray(X[i]) for i in range(n if n is not None else len(X))])
    X = torch.Tensor(X_numpy).unsqueeze(1)

    which = np.random.choice(len(y_numpy), int((1-semisupervised_proportion)*len(y_numpy)), replace = False)
    y_for_verification = copy.deepcopy(y_numpy)
    y_numpy[which] = -1

    e.fit(X, y_numpy, y_for_verification = y_for_verification, plot = True)
