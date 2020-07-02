from pathlib import Path

import numpy as np
from scipy.spatial import distance
from sklearn import datasets


def get_distances(x, distance_name):
    distances = distance.squareform(distance.pdist(x, metric=distance_name))
    return distances

def sbss(x, y, n_splits, validation_rate_from_train=0.0, distance_name='euclidean'):
    # Default checking from scikitlearn kfold
    _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
    # y_inv encodes y according to lexicographic order. We invert y_idx to
    # map the classes so that they are encoded by order of appearance:
    # 0 represents the first label appearing in y, 1 the second, etc.
    _, class_perm = np.unique(y_idx, return_inverse=True)
    y_encoded = class_perm[y_inv]

    n_classes = len(y_idx)
    y_counts = np.bincount(y_encoded)
    min_groups = np.min(y_counts)
    if np.all(n_splits > y_counts):
        raise ValueError("n_splits=%d cannot be greater than the"
                         " number of members in each class."
                         % (n_splits))
    if n_splits > min_groups:
        print(("The least populated class in y has only %d"
                        " members, which is less than n_splits=%d."
                        % (min_groups, n_splits)))
        # test_folds = np.empty(len(self.y), dtype='i')

    ################### START SBSS ###################
    train_index = None
    val_index = None
    test_index = None
    data = {'train': {}, 'test': {}}

    distances = get_distances(x, distance_name)

    used_indexes = np.zeros(len(y)).astype(np.bool)
    # cada append tera tamanho = n_split
    folds_list = [[] for _ in range(n_splits)]  # array with k rows and n_samples cols. Each column belongs to the same class
    splits = np.arange(n_splits)

    fold_col_lb = []

    for k in range(n_classes):
        idx_k = y.squeeze() == k

        to_split_idxs_lb = idx_k.sum()

        while to_split_idxs_lb >= n_splits:
            # Pegando o elemento pivo, aquele que possui a menor distancia dele para os indicis ainda nao utilizados que fazem parte da classe k
            # Usando (i) apenas os indices nao utilizados anteriormente e (ii) apenas da classe k.
            considered_idxs = (~used_indexes) & idx_k
            # Pega os n_splits exemplos de menor distancia entre si para fazer parte dos splits
            sum_distances = distances[:, considered_idxs].sum(1)
            # Colocando np.inf para forcar a ida dos indices que nao fazem sentido ficar na ultima posicao do sort.
            sum_distances[~considered_idxs] = np.inf
            # Pega o sample pivo, que tem a menor distancia para todos os outros samples da mesma classe
            pivot_idx = np.argpartition(sum_distances, 0)[0]

            used_indexes[pivot_idx] = True

            nearby_samples = [pivot_idx]

            for split_idx in splits[1:]:
                # Distancia entre todos os elementos e os samples escolhidos para compor os splits correntes
                sum_distances = distances[:, nearby_samples].sum(1)
                # Ignorando indices que ja foram usados ou que nao sao da classe k
                sum_distances[~considered_idxs] = np.inf
                # Ignorando o proprio pivo, ja que nao faz sentido comparar a distancia dele com ele mesmo
                sum_distances[pivot_idx] = np.inf

                # Pega o indice de menor valor (elemento mais proximo)
                closest_sample_idx = np.argpartition(sum_distances, 0)[0]
                nearby_samples.append(closest_sample_idx)

                used_indexes[closest_sample_idx] = True

                # Desconsiderando os indices para as proximas iteracoes
                considered_idxs[closest_sample_idx] = False

            fold_col_lb.append(k)

            # Embaralha os samples para garantir a estocasticidade
            np.random.shuffle(nearby_samples)

            for split_idx in splits:
                folds_list[split_idx].append(nearby_samples[split_idx])

            to_split_idxs_lb = to_split_idxs_lb - n_splits

    folds = np.array(folds_list)
    fold_col_lb = np.array(fold_col_lb)
    for split in range(n_splits):
        train_splits = np.ones(n_splits).astype(np.bool)
        train_splits[split] = False

        test_index = folds[split]
        data['test']['data'] = x[test_index]
        data['test']['target'] = y[test_index]

        train_val_index = folds[train_splits, :].ravel()
        if validation_rate_from_train is None or validation_rate_from_train == 0:
            data['train']['data'] = x[train_val_index]
            data['train']['target'] = y[train_val_index]
            train_index = train_val_index
        else:
            val = []
            for k in range(n_classes):
                idx_k = y[train_val_index].squeeze() == k
                n_validations = int(idx_k.sum() * validation_rate_from_train)
                folds__lb_k = folds[train_splits][:, fold_col_lb == k]
                nb_train_folds, nb_k_in_fold = folds__lb_k.shape
                nb_used_entire_folds = int(n_validations / nb_k_in_fold)

                # Uma vez que cada fold (linha) tem samples shuffled do mesmo label (coluna), pegar os primeiros K nao eh um problema
                tmp_val = []
                tmp_val.extend(folds__lb_k[:nb_used_entire_folds, :].ravel())

                missing_samples = n_validations - len(tmp_val)
                tmp_val.extend(folds__lb_k[nb_used_entire_folds, :missing_samples].ravel())

                val.extend(tmp_val)

            val_index = np.array(val).ravel()
            train_index = np.array(list(set(train_val_index.ravel()) - set(val_index)))

            data['train']['data'] = x[train_index]
            data['train']['target'] = y[train_index]
            data['val'] = {}
            data['val']['data'] = x[val_index]
            data['val']['target'] = y[val_index]

        yield train_index, val_index, test_index, data


def example():
    # iris = datasets.load_iris()
    # x = iris.data
    # y = iris.target

    data = np.array(
        [
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [11, 11, 11, 1],
            [0, 11, 11, 1],
            [0, 0, 11, 1],
            [111, 111, 111, 1],
            [0, 111, 111, 1],
            [0, 0, 111, 1],

            [2, 2, 2, 0],
            [0, 2, 2, 0],
            [0, 0, 2, 0],
            [22, 22, 22, 0],
            [0, 22, 22, 0],
            [0, 0, 22, 0],
            [222, 222, 222, 0],
            [0, 222, 222, 0],
            [0, 0, 222, 0],
        ]
    )

    x = data[:, :-1]
    y = data[:, -1]

    for i, (train_index, val_index, test_index, data) in enumerate(sbss(x=x, y=y, n_splits=3, validation_rate_from_train=0.50)):
        print(f'Split {i}, train {train_index}, val {val_index}, test {test_index}')
        x_train = data['train']['data']
        x_test = data['test']['data']
        y_train = data['train']['target']
        y_test = data['test']['target']

        train_matrix = np.concatenate((x_train, y_train[:, None]), axis=1)
        test_matrix = np.concatenate((x_test, y_test[:, None]), axis=1)

        print(f'Train:\n{train_matrix}')
        print(f'Test:\n{test_matrix}')

        if val_index is not None:
            x_val = data['val']['data']
            y_val = data['val']['target']

            val_matrix = np.concatenate((x_val, y_val[:, None]), axis=1)

            print(f'Val:\n{val_matrix}')


        print('-'*80)

    print('Pode-se perceber que os conjuntos sempre possuem as representacoes de unidades, dezenas e centenas de cada classe 0 e 1.')
    print('Recomendo que os dados estejam normalizados. Coloquei valores altos so para demonstrar o funcionamento mais facilmente.')

if __name__ == '__main__':
    example()