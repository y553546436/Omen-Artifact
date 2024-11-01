import numpy as np
import torch
from partial import filter_candidate

def get_dists(enc_x, class_hvs, binary):
    n, dim = enc_x.shape
    classes = class_hvs.shape[0]
    if binary:
        dists = torch.sum(torch.abs(class_hvs.unsqueeze(1) - enc_x.unsqueeze(0)), dim=2) / dim
    else:
        dists = 1 - torch.nn.functional.cosine_similarity(class_hvs.unsqueeze(1), enc_x.unsqueeze(0), dim=2)
    return dists


def batch_get_sum_dists(enc_x, y, class_hvs, binary):
    n, dim = enc_x.shape
    classes = class_hvs.shape[0]
    dists = get_dists(enc_x, class_hvs, binary)
    y_pred = torch.argmin(dists, dim=0)
    # mask is a (classes, n) matrix, where mask[i, j] = 1 if y[j] == i (sample j is in class i), otherwise 0
    mask = y.unsqueeze(0) == torch.arange(classes, device=y.device).unsqueeze(1)
    # correct_mask is a (classes, n) matrix, where correct_mask[i, j] = 1 if y[j] == i (sample j is in class i) and y_pred[j] == i (sample j is classified as class i), otherwise 0
    correct_mask = mask & (y.unsqueeze(0) == y_pred.unsqueeze(0))
    correct_dists = dists * correct_mask.int().float()
    incorrect_dists = dists * (mask.int() - correct_mask.int()).float()
    return correct_mask, correct_dists.sum(dim=1), torch.square(correct_dists).sum(dim=1), incorrect_dists.sum(dim=1), torch.square(incorrect_dists).sum(dim=1)


# threshold for class t is calculated as the mean of (1) mean+std of distances from training samples in class t to class hypervector t, and (2) mean-std of distances from the training samples not in class t to class hypervector t, where std is the standard deviation.
def get_mean_thresholds(enc_x, y, class_hvs, binary):
    # enc_x: (n, dim), y (n,) labels for n samples, class_hvs: (classes, dim) torch tensors
    # returns: (classes, ) numpy array
    n, dim = enc_x.shape
    classes = class_hvs.shape[0]
    mask = y.unsqueeze(0) == torch.arange(classes, device=y.device).unsqueeze(1)
    # print(f'mask: {mask}')
    # mask shape (classes, n)
    correct_dists = torch.zeros(classes, device=enc_x.device)
    correct_sq_dists = torch.zeros(classes, device=enc_x.device)
    incorrect_dists = torch.zeros(classes, device=enc_x.device)
    incorrect_sq_dists = torch.zeros(classes, device=enc_x.device)
    batch_size = 1000
    correct_mask_list = []
    for i in range(0, n, batch_size):
        correct_mask_batch, correct_dists_batch, correct_sq_dists_batch, incorrect_dists_batch, incorrect_sq_dists_batch = batch_get_sum_dists(enc_x[i:i+batch_size], y[i:i+batch_size], class_hvs, binary)
        correct_dists += correct_dists_batch.to(correct_dists.device)
        correct_sq_dists += correct_sq_dists_batch.to(correct_dists.device)
        incorrect_dists += incorrect_dists_batch.to(correct_dists.device)
        incorrect_sq_dists += incorrect_sq_dists_batch.to(correct_dists.device)
        correct_mask_list.append(correct_mask_batch)
    correct_mask = torch.cat(correct_mask_list, dim=1)
    correct_mean = correct_dists / correct_mask.sum(dim=1).float()
    correct_std = torch.sqrt(correct_sq_dists / correct_mask.sum(dim=1).float() - torch.square(correct_mean))
    incorrect_mean = incorrect_dists / (mask.sum(dim=1) - correct_mask.sum(dim=1)).float()
    incorrect_std = torch.sqrt(incorrect_sq_dists / (mask.sum(dim=1) - correct_mask.sum(dim=1)).float() - torch.square(incorrect_mean))
    thresholds = (correct_mean + correct_std + incorrect_mean - incorrect_std) / 2
    return thresholds.cpu().numpy()


def get_similarities(enc_x, model, binary):
    if not binary: # cosine similarity
        diss = torch.cumsum(enc_x.unsqueeze(1) * model, dim=-1)
        similarity = diss / (torch.sqrt(torch.cumsum(torch.square(enc_x), dim=-1)).unsqueeze(1) * torch.sqrt(torch.cumsum(torch.square(model), dim=-1)).unsqueeze(0))
    else: # hamming distance
        diss = torch.cumsum(enc_x.unsqueeze(1) != model, dim=-1).double()
        dim_indices = torch.arange(1, model.shape[-1] + 1, device=model.device)
        similarity = 1 - diss / dim_indices
    return similarity


def batch_get_acc(enc_x, y, class_hvs, binary, mode, threshold):
    n, dim = enc_x.shape
    classes = class_hvs.shape[0]
    similarity = get_similarities(enc_x, class_hvs, binary)
    cand = torch.argmax(similarity, dim=1)
    if mode == 'diff':
        similarity_diff = torch.abs(similarity.unsqueeze(2) - similarity.unsqueeze(1))
        similarity_diff = filter_candidate(similarity_diff, cand)
    if dim == 256:
        start, freq = 16, 4
    else:
        if binary:
            start, freq = 512, 64
        else:
            start, freq = 128, 64
    test_locs = torch.arange(start-1, dim, freq, device=enc_x.device)
    if test_locs[-1] != dim - 1:
        test_locs = torch.cat((test_locs, torch.tensor([dim-1], device=enc_x.device)))
    if mode == 'diff':
        confidence = torch.all(similarity_diff > threshold, dim=-1)
    else:
        confidence = torch.max(similarity, dim=1).values > threshold
    confidence[:, dim-1] = True
    stop_indices = torch.argmax(confidence[:, test_locs].int(), axis=1)
    stop_locs = test_locs[stop_indices]
    correct_batch = sum(cand[torch.arange(n), stop_locs] == y)
    return correct_batch


def binary_search_threshold(enc_x, y, class_hvs, binary, mode):
    batch_size = 100
    n, dim = enc_x.shape
    classes = class_hvs.shape[0]
    total_correct = 0
    for i in range(0, n, batch_size):
        total_correct += batch_get_acc(enc_x[i:i+batch_size], y[i:i+batch_size], class_hvs, binary, 'absolute', 100) # large threshold, all samples are not confident
    orig_acc = total_correct / n
    max_acc_drop = 0.01
    print(f'Original accuracy: {orig_acc}')

    # binary search the threshold
    threshold_low, threshold_high = 0, 1
    while threshold_high - threshold_low > 1e-6:
        threshold = (threshold_low + threshold_high) / 2
        correct = 0
        for i in range(0, n, batch_size):
            correct += batch_get_acc(enc_x[i:i+batch_size], y[i:i+batch_size], class_hvs, binary, mode, threshold)
        acc = correct / n
        if acc < orig_acc - max_acc_drop:
            threshold_low = threshold
        else:
            threshold_high = threshold
        print(f'Threshold: {threshold}, Accuracy: {acc}')
    return threshold


# confident when distance similarity between cand and second candidate is greater than threshold
def get_diff_threshold(enc_x, y, class_hvs, binary):
    return binary_search_threshold(enc_x, y, class_hvs, binary, 'diff')


def get_absolute_threshold(enc_x, y, class_hvs, binary):
    return binary_search_threshold(enc_x, y, class_hvs, binary, 'absolute')


if __name__ == '__main__':
    # example usage
    n, dim = 3, 10
    classes = 2
    binary = True
    enc_x = torch.randint(0, 2, (n, dim))
    y = torch.randint(0, classes, (n,))
    class_hvs = torch.randint(0, 2, (classes, dim))
    print(f'class hv: {class_hvs}')
    print(f'encodings: {enc_x}')
    print(f'labels: {y}')
    thresholds = get_mean_thresholds(enc_x, y, class_hvs, binary)
    print(f'thresholds: {thresholds}')
