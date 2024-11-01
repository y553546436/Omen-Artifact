import torch
import numpy as np
import pickle
from tqdm import tqdm
import os


def filter_candidate(tensor, cand):
    n, m, d = tensor.shape[0], tensor.shape[1], tensor.shape[3]
    n_indices = torch.arange(n).unsqueeze(1).expand(-1, d).reshape(-1)
    d_indices = torch.arange(d).unsqueeze(0).expand(n, -1).reshape(-1)
    cand_flat = cand.reshape(-1)

    selected = tensor[n_indices, cand_flat, :, d_indices]
    # Create a mask that is True for all indices except those in cand
    mask = torch.ones(n, d, m, dtype=torch.bool)
    mask[torch.arange(n).unsqueeze(1).expand_as(cand), torch.arange(d).unsqueeze(0).expand_as(cand), cand] = 0
    # selected = selected.reshape(m, n, d)
    return selected.reshape(n, d, m)[mask].reshape(n, d, m - 1)


def stats_test(testvec: torch.Tensor, testlbl: torch.Tensor, model: torch.Tensor, alpha: float, offset: int = 64, batch_size: int = 64, binary: bool = False):
    if not binary: # dot product
        diss = torch.cumsum(testvec.unsqueeze(1) * model, dim=-1)
    else: # hamming distance
        diss = torch.cumsum(testvec.unsqueeze(1) != model, dim=-1).float()

    testvec_sq = torch.square(testvec)
    # test_norm = torch.sqrt(torch.cumsum(testvec_sq, dim=-1))
    # model_norm = torch.sqrt(torch.cumsum(torch.square(model), dim=-1))
    if not binary:
        model_pair_diff = model.unsqueeze(1) - model.unsqueeze(0)
        diff2 = torch.cumsum(torch.square(model_pair_diff) * testvec_sq.unsqueeze(1).unsqueeze(1), dim=-1)
    else:
        model_pair_diff = model.unsqueeze(1) != model.unsqueeze(0)
        diff2 = torch.cumsum(model_pair_diff.unsqueeze(0).expand(testvec_sq.shape[0], -1, -1, -1), dim=-1).float()

    # dimension index vector
    dim_indices = torch.arange(1, model.shape[-1] + 1, device=model.device)
    # cand matrix to track winner change
    if not binary:
        cand = torch.argmax(diss, dim=1)
    else:
        cand = torch.argmin(diss, dim=1)
    # paired differences between different class for diss
    delta_hat = torch.abs(diss.unsqueeze(2) - diss.unsqueeze(1)) / dim_indices
    delta_hat = filter_candidate(delta_hat, cand)
    diff2_n = diff2 / dim_indices
    diff2_n = filter_candidate(diff2_n, cand)

    SE_hat = torch.sqrt((diff2_n - torch.square(delta_hat)) / dim_indices.unsqueeze(1))
    W = delta_hat / SE_hat

    # p_values = (1 - norm.cdf(torch.abs(W)))
    p_values = 1 - 0.5 * (1 + torch.erf(torch.abs(W) / np.sqrt(2)))

    sorted_p_values, _ = torch.sort(p_values, dim=-1)

    m, d = model.shape[0], model.shape[1]
    indices = torch.arange(m - 1, device=p_values.device)[None, None, :]
    mask = sorted_p_values <= (alpha / (m - 1 - indices))
    confidence = torch.all(mask, dim=-1)

    # create a mask to test confidence only after offset and at the end of batch_size intervals
    mask = torch.zeros_like(confidence)
    mask[:, (offset-1)::batch_size] = 1
    mask[:, -1] = 1
    confidence_batched = confidence & mask

    # find the indices of first True value in each row of confidence
    first_true_indices = torch.argmax(confidence_batched.int(), dim=-1)
    # correct the indices where there is no True value
    all_false_mask = ~torch.any(confidence_batched, dim=-1)
    first_true_indices[all_false_mask] = d - 1
    predictions = cand[torch.arange(cand.shape[0]), first_true_indices]
    res = predictions == testlbl
    correct = torch.sum(res)
    total = res.shape[0]
    return correct, total, cand, confidence, p_values


def other_strategies(testvec: torch.Tensor, testlbl: torch.Tensor, model: torch.Tensor, threshold: float, strategy: str, binary: bool, mean_thresholds=None):
    if not binary: # cosine similarity
        diss = torch.cumsum(testvec.unsqueeze(1) * model, dim=-1)
        # cand = torch.argmax(diss, dim=1)
        # divide by norms
        similarity = diss / (torch.sqrt(torch.cumsum(torch.square(testvec), dim=-1)).unsqueeze(1) * torch.sqrt(torch.cumsum(torch.square(model), dim=-1)).unsqueeze(0))
    else: # hamming distance
        diss = torch.cumsum(testvec.unsqueeze(1) != model, dim=-1).double()
        # cand = torch.argmin(diss, dim=1)
        dim_indices = torch.arange(1, model.shape[-1] + 1, device=model.device)
        similarity = 1 - diss / dim_indices
    cand = torch.argmax(similarity, dim=1)
    if strategy == 'diff':
        # confident when distance similarity between cand and second candidate is greater than threshold
        similarity_diff = torch.abs(similarity.unsqueeze(2) - similarity.unsqueeze(1))
        # print(f'similarity_diff: {similarity_diff.shape}')
        similarity_diff = filter_candidate(similarity_diff, cand)
        # print(f'similarity_diff: {similarity_diff.shape}')
        confidence = torch.all(similarity_diff > threshold, dim=-1)
        # print(f'confidence: {confidence.shape}')
    elif strategy == 'absolute':
        # confident when maximum similarity is greater than threshold
        confidence = torch.max(similarity, dim=1).values > threshold
    elif strategy == 'mean':
        # confident when winner class' similarity exceeds the threshold derived from training data
        thresholds = torch.from_numpy(mean_thresholds).to(device) # shape: (classes,)
        thresholds = thresholds.unsqueeze(0).expand(n, -1).unsqueeze(1).expand(-1, cand.shape[-1], -1)
        # cand shape (n, dim), similarity shape (n, classes, dim), thresholds shape (n, dim, classes)
        # print(f'similarity: {similarity.shape}, thresholds: {thresholds.shape}')
        confidence = 1 - torch.max(similarity, dim=1).values > thresholds.gather(2, cand.unsqueeze(2)).squeeze(2)
    return cand, confidence


def load_and_test(data_dir: str, output_dir: str, alpha: float, binary: bool, args):
    # load in the trained model and test data
    with open(f'{data_dir}/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{data_dir}/testvec.pkl', 'rb') as f:
        testvec = pickle.load(f)
    with open(f'{data_dir}/testlbl.pkl', 'rb') as f:
        testlbl = pickle.load(f)

    # init torch and device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f'Using NVIDIA GPU: {torch.cuda.get_device_name()}.')
    else:
        device = 'cpu'
        print('Using CPU.')
    model = torch.from_numpy(model).to(device)
    testvec = torch.from_numpy(testvec).to(device)
    testlbl = torch.from_numpy(testlbl).to(device)
    print('Model and test data loaded.')
    print(f'Model shape: {model.size()}')
    print(f'Test vector shape: {testvec.size()}')
    print(f'Test label shape: {testlbl.size()}')

    np.random.seed(0)
    torch.manual_seed(0)
    if args.ber != 0:
        print(f'Bit Error Rate: {args.ber}')
        testvec_bitflip = torch.bernoulli(torch.full_like(testvec, args.ber, dtype=torch.float)).int().to(device)
        testvec = testvec ^ testvec_bitflip
        print('Test vector bitflipped.')
        model_bitflip = torch.bernoulli(torch.full_like(model, args.ber, dtype=torch.float)).int().to(device)
        model = model ^ model_bitflip

    n = testvec.size(0)
    batch_size = 100
    if args.strategy == 'omen':
        correct = 0
        total = 0
        cands = []
        confidence = []
        p_values = []
        for i in tqdm(range(0, n, batch_size)):
            testvec_batch = testvec[i:i + batch_size]
            testlbl_batch = testlbl[i:i + batch_size]
            correct_batch, total_batch, cand, confidence_batch, p_values_batch = stats_test(testvec_batch, testlbl_batch, model, alpha, 64, binary=binary)
            correct += correct_batch
            total += total_batch
            cands.append(cand)
            confidence.append(confidence_batch)
            p_values.append(p_values_batch)
        cand = torch.cat(cands, dim=0)
        confidence = torch.cat(confidence, dim=0)
        p_values = torch.cat(p_values, dim=0)


        # ================== DEBUG ==================
        print('Finding examples that are classified correctly using all dimensions, but misclassified using 9216 dimensions.')
        lang_labels = ['dutch', 'english', 'franch', 'german', 'italian']
        # checkpoints = torch.array([1024, 2048, 3072, 4096, 5120, 7168]) - 1
        checkpoints = torch.from_numpy(np.array([1024, 2048, 4096, 9216, 10048]) - 1).to(device)
        load = getattr(__import__('language'), 'load')
        x, x_test, y, y_test = load()
        '''
        mask = cand[:, -1] == testlbl
        mask2 = cand[:, 9216] != testlbl
        mask = mask & mask2
        for index in range(n):
            if mask[index]:
                diss = torch.cumsum(testvec[index].unsqueeze(0) != model, dim=-1)
                diss = diss / torch.arange(1, diss.shape[-1] + 1, device=diss.device)
                diss = torch.transpose(diss, 0, 1)
                print(f'sentence {index}: {x_test[index]}')
                print(f'correct label: {lang_labels[y_test[index]]}, dist in last dimension: {diss[-1, :]}')
                print(f'misclassified label: {lang_labels[cand[index, 9216]]}, dist when misclassified: {diss[9216, :]}')
                print(f'labels at checkpoints: {[lang_labels[c] for c in cand[index, checkpoints]]}')
                print(f'distances at checkpoints: {diss[checkpoints, :]}')
                print(f'p-values at checkpoints: {p_values[index, checkpoints]}')
                print(f'confidence at checkpoints: {confidence[index, checkpoints]}')

        print('==============================================================')

        print('Finding examples that are classified correctly using just 2048 dimensions and confident at 2048th dimension.')
        mask = cand[:, -1] == testlbl
        mask2 = cand[:, 2047] == testlbl
        mask3 = confidence[:, 2047]
        mask = mask & mask2 & mask3
        indices = torch.arange(n)[mask]
        # indices = indices[torch.randperm(indices.size(0))[:10]]
        for index in indices:
            if lang_labels[y_test[index]] != 'dutch':
                continue
            diss = torch.cumsum(testvec[index].unsqueeze(0) != model, dim=-1)
            diss = diss / torch.arange(1, diss.shape[-1] + 1, device=diss.device)
            diss = torch.transpose(diss, 0, 1)
            print(f'sentence {index}: {x_test[index]}')
            print(f'correct label: {lang_labels[y_test[index]]}, dist in last dimension: {diss[-1, :]}')
            print(f'classified label: {lang_labels[cand[index, 2047]]}, dist in 2048th dimension: {diss[2047, :]}')
            print(f'last 9216 dimension: {lang_labels[cand[index, 9216]]}, dist in last 1000 dimension: {diss[9216, :]}')
            print(f'labels at checkpoints: {[lang_labels[c] for c in cand[index, checkpoints]]}')
            print(f'distances at checkpoints: {diss[checkpoints, :]}')
            print(f'p-values at checkpoints: {p_values[index, checkpoints]}')
            print(f'confidence at checkpoints: {confidence[index, checkpoints]}')

        '''
        print('==============================================================')
        # two examples are sentences 9 and 383
        for index in [383, 9]:
            diss = torch.cumsum(testvec[index].unsqueeze(0) != model, dim=-1)
            diss = diss / torch.arange(1, diss.shape[-1] + 1, device=diss.device)
            diss = torch.transpose(diss, 0, 1)
            print(f'sentence {index}: {x_test[index]}')
            print(f'correct label: {lang_labels[y_test[index]]}, dist in last dimension: {diss[-1, :]}')
            # print(f'classified label: {lang_labels[cand[index, 2047]]}, dist in 2048th dimension: {diss[2047, :]}')
            print(f'last 1000 dimension: {lang_labels[cand[index, -1000]]}, dist in last 1000 dimension: {diss[-1000, :]}')
            print(f'labels at checkpoints: {[lang_labels[c] for c in cand[index, checkpoints]]}')
            print(f'distances at checkpoints: {diss[checkpoints, :]}')
            print(f'p-values at checkpoints: {p_values[index, checkpoints]}')
            print(f'confidence at checkpoints: {confidence[index, checkpoints]}')


        # ================== DEBUG ==================

        accuracy = correct / total
        print(f'Accuracy: {accuracy}')
    else:
        threshold = args.threshold
        if args.strategy == 'mean':
            with open(f'{data_dir}/mean_thresholds.pkl', 'rb') as f:
                mean_thresholds = pickle.load(f) # mean_threesholds are distance thresholds
        else:
            mean_thresholds = None
        cands = []
        confidences = []
        for i in tqdm(range(0, n, batch_size)):
            testvec_batch = testvec[i:i + batch_size]
            testlbl_batch = testlbl[i:i + batch_size]
            cand, confidence = other_strategies(testvec_batch, testlbl_batch, model, threshold, args.strategy, binary, mean_thresholds)
            cands.append(cand)
            confidences.append(confidence)
        cand = torch.cat(cands, dim=0)
        confidence = torch.cat(confidences, dim=0)


    print(f'candidate changes: {cand.unique(return_counts=True, dim=-1)}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f'{output_dir}/cand.pkl', 'wb') as f:
        pickle.dump(cand, f)
    print('Candidate matrix saved.')
    with open(f'{output_dir}/confidence.pkl', 'wb') as f:
        pickle.dump(confidence, f)
    print('Confidence matrix saved.')

    if args.strategy == 'omen':
        with open(f'{output_dir}/p_values.pkl', 'wb') as f:
            pickle.dump(p_values, f)
        print('P-values saved.')

    print('Test finished.')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load and test model with Omen')
    parser.add_argument('--dataset', type=str, help='dataset to use')
    parser.add_argument('--data', type=str, help='Directory containing model and test data')
    parser.add_argument('--output', type=str, help='Directory to save output')
    parser.add_argument('--alpha', help="Confidence Threshold", type=float, default=0.05)
    parser.add_argument('-b', "--binary", help="Use binary hypervectors", action='store_true')
    parser.add_argument('--strategy', type=str, help='Strategy to use', default='omen') # omen, diff, absolute, mean
    parser.add_argument('--threshold', type=float, help='threshold values for diff and absolute')
    parser.add_argument('--ber', type=float, help='Bit Error Rate for binary hypervectors', default=0)
    args = parser.parse_args()
    if args.data is None and args.dataset is not None:
        args.data = f'data-{args.dataset}'
    if args.output is None and args.dataset is not None:
        args.output = f'output-{args.dataset}'
    if args.data is None and args.dataset is None:
        print('Using default data directory: data')
        args.data = 'data'
    if args.output is None and args.dataset is None:
        print('Using default output directory: output')
        args.output = 'output'
    if args.strategy != 'omen':
        assert args.strategy in ['diff', 'absolute', 'mean'], 'Invalid strategy.'
        if args.strategy != 'mean':
            assert args.threshold is not None, 'Threshold required for diff and absolute strategy.'
    if args.ber != 0:
        assert args.binary, 'Bit Error Rate only works with binary hypervectors.'
    print(f'Loading model and test data from {args.data} and saving output to {args.output}.')
    load_and_test(args.data, args.output, args.alpha, True if args.binary else False, args)