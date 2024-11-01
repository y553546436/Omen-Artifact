#include "model.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <chrono>

#ifdef BINARY
typedef uint64_t hv_type;
typedef int dis_type;
#else // real-valued
typedef double hv_type;
typedef double dis_type;
#endif

#include "testdata.h"

static_assert(NUMFEATURE == DATA_NUMFEATURE, "Defined model NUMFEATURE does not match the data NUMFEATURE");
static_assert(NUMCLASS == DATA_NUMCLASS, "Defined model NUMCLASS does not match the data NUMCLASS");


/*
 * All capitalized variables and macros are constants defined in hdc.h which is generated from a Python script
 */

#ifdef LANGUAGE
// get the index of a character in the codebook
int get_code_index(int c) {
    return std::lower_bound(CHAR_MAP, CHAR_MAP + NUMCHAR, (int)(c)) - CHAR_MAP;
}
#endif

#ifdef BLDC
float vb_hidden_layer[NUMFEATURE][VALUE_BOX_HIDDEN_DIM];
hv_type vb_output_layer[NUMFEATURE][VALUE_HV_DIM];
int current_test_no = -1;


void ENCODE_INIT(int test_no) {
    current_test_no = test_no;
    for (int i = 0; i < NUMFEATURE; i++) {
        for (int j = 0; j < VALUE_BOX_HIDDEN_DIM; j++) {
            vb_hidden_layer[i][j] = x[test_no][i] * value_box_fc1_weight[j] + value_box_fc1_bias[j];
            // batch normalization
            vb_hidden_layer[i][j] -= vb_bn_running_mean[j];
            vb_hidden_layer[i][j] /= vb_bn_running_std[j];
            vb_hidden_layer[i][j] = vb_hidden_layer[i][j] * vb_bn_gamma[j] + vb_bn_beta[j];
            // activation
            vb_hidden_layer[i][j] = std::tanh(vb_hidden_layer[i][j]);
        }
    }
}

hv_type ENCODE(int dim) {
    hv_type value = 0;
    if (dim < VALUE_HV_DIM) {
        for (int i = 0; i < NUMFEATURE; i++) {
            float output = 0;
            for (int j = 0; j < VALUE_BOX_HIDDEN_DIM; j++) {
                output += vb_hidden_layer[i][j] * value_box_fc3_weight[dim][j];
            }
            vb_output_layer[i][dim] = output < 0;
            value += vb_output_layer[i][dim] ^ feature_layer_weight[dim][i];
        }
    } else {
        for (int i = 0; i < NUMFEATURE; i++) {
            value += vb_output_layer[i][dim % VALUE_HV_DIM] ^ feature_layer_weight[dim][i];
        }
    }
    return value >= 0.5 * NUMFEATURE;
}

// encode all dimensions of a test
hv_type *ENCODE_ALL(int test_no) {
    ENCODE_INIT(test_no);
    hv_type *hv = new hv_type[FEATURE_HV_DIM];
    for (int i = 0; i < FEATURE_HV_DIM; i++) {
        hv[i] = ENCODE(i);
    }
    return hv;
}

// encode chunk of dimensions of a test
hv_type *ENCODE_CHUNK(int dim_start, int dim_end) {
    hv_type *hv = new hv_type[dim_end - dim_start];
    for (int i = dim_start; i < dim_end; i++) {
        hv[i - dim_start] = ENCODE(i);
    }
    return hv;
}

// normal inference without omen
void normal(int &correct) {
    correct = 0;
    int total = 0;
    for (int test_no = 0; test_no < DATA_NUMTEST; test_no++) {
        dis_type dis[NUMCLASS];
        hv_type *hv = ENCODE_ALL(test_no);
        // printf("Encoded\n");
        for (int class_no = 0; class_no < NUMCLASS; class_no++) {
            dis[class_no] = 0;
            for (int dim = 0; dim < FEATURE_HV_DIM; dim++) {
                dis[class_no] += hv[dim] ^ class_layer_weight[dim][class_no];
            }
        }
        int result = std::min_element(dis, dis + NUMCLASS) - dis;
        delete[] hv;
        correct += (result == y[test_no]);
        total++;
    }
    // printf("Accuracy: %f (%d / %d)\n", (float) correct / total, correct, total);
}

void omen(int &dims, int &correct) {
    correct = 0;
    dims = 0;
    int total = 0;
    for (int test_no = 0; test_no < DATA_NUMTEST; test_no++) {
        dis_type dis[NUMCLASS];
        std::memset(dis, 0, sizeof(dis));
        ENCODE_INIT(test_no);
        for (int dim_start = 0, dim_end = FREQ; dim_start < FEATURE_HV_DIM; dim_start = dim_end, dim_end = std::min(dim_start + FREQ, FEATURE_HV_DIM)) {
            hv_type *hv = ENCODE_CHUNK(dim_start, dim_end);
            for (int dim = 0; dim < dim_end - dim_start; dim++) {
                int global_dim = dim_start + dim;
                hv_type value = hv[dim];
                for (int class_no = 0; class_no < NUMCLASS; class_no++) {
                    dis[class_no] += value ^ class_layer_weight[global_dim][class_no];
                }
            }
            delete[] hv;
            if (dim_end < START) {
                continue;
            }
            int cand = std::min_element(dis, dis + NUMCLASS) - dis;
            // statistical tests with precomputed thresholds
            float W_squared[NUMCLASS];
            for (int class_no = 0; class_no < NUMCLASS; class_no++) {
                if (class_no == cand) {
                    W_squared[class_no] = 1e9;
                } else {
                    double tmp = (dis[class_no]-dis[cand]);
                    tmp *= tmp;
                    // precomputed diff2 for binary
                    W_squared[class_no] = (tmp * (dim_end<<6)) / (DIFF2[cand][dim_start/FREQ][class_no] * (dim_end<<6) - tmp);
                }
            }
            std::sort(W_squared, W_squared + NUMCLASS);
            bool flag = true;
            for (int class_no = 0; class_no < NUMCLASS-1; class_no++) {
                if (W_squared[class_no] < THRESHOLD[class_no]) { // THRESHOLD shape is (NUMCLASS-1)
                    flag = false;
                }
            }
            if (flag || dim_end == FEATURE_HV_DIM) { // pass all tests, return the class with maximum dot product similarity
                correct += (cand == y[test_no]);
                total++;
                dims += dim_end;
                break;
            }
        }
    }
    // printf("Accuracy: %f (%d / %d)\n", (float) correct / total, correct, total);
}

#else

// encode a test in a specific dimension
hv_type ENCODE(int test_no, int dim) {
    hv_type hv = 0;
#ifdef REAL
    for (int feature_no = 0; feature_no < NUMFEATURE; feature_no++) {
        hv += GAUSSIAN[feature_no][dim] * x[test_no][feature_no];
    }
    hv = std::sin(hv) * std::cos(hv + OFFSET[dim]);
    return hv;
#else
    int cnts[64];
    std::memset(cnts, 0, sizeof(cnts));
#ifdef LANGUAGE
    for (int trigram_start = 1; trigram_start < x[test_no][0] - 2 + 1; trigram_start++) {
        uint64_t trigram = 0;
        for (int i = 0; i < 3; i++) {
            trigram ^= CHAR_CODEBOOK[get_code_index(x[test_no][trigram_start + i])][(dim+i)%NUMDIM];
        }
        for (int i = 0; i < 64; i++) {
            cnts[i] += (trigram >> i) & 1;
        }
    }
    for (int i = 0; i < 64; i++) {
        hv |= (hv_type)(cnts[i] > (x[test_no][0] - 2) / 2) << i;
    }
#else
    for (int feature_no = 0; feature_no < NUMFEATURE; feature_no++) {
        int level = std::ceil((x[test_no][feature_no] - MIN_VAL) * LEVELS / (MAX_VAL - MIN_VAL));
        uint64_t bind = CODEBOOK[level][dim] ^ BASIS[feature_no][dim];
        for (int i = 0; i < 64; i++) {
            cnts[i] += (bind >> i) & 1;
        }
    }
    for (int i = 0; i < 64; i++) {
        hv |= (hv_type)(cnts[i] > NUMFEATURE / 2) << i;
    }
#endif
    return hv;
#endif
}

// encode all dimensions of a test
hv_type *ENCODE_ALL(int test_no) {
    hv_type *hv = new hv_type[NUMDIM];
    for (int dim = 0; dim < NUMDIM; dim++) {
        hv[dim] = ENCODE(test_no, dim);
    }
    return hv;
}

// encode chunk of dimensions of a test
hv_type *ENCODE_CHUNK(int test_no, int dim_start, int dim_end) {
    hv_type *hv = new hv_type[dim_end - dim_start];
    for (int dim = dim_start; dim < dim_end; dim++) {
        hv[dim - dim_start] = ENCODE(test_no, dim);
    }
    return hv;
}

// normal inference without omen
void normal(int &correct) {
    correct = 0;
    int total = 0;
    for (int test_no = 0; test_no < DATA_NUMTEST; test_no++) {
        dis_type dis[NUMCLASS];
        hv_type *hv = ENCODE_ALL(test_no);
        for (int class_no = 0; class_no < NUMCLASS; class_no++) {
            dis[class_no] = 0;
            for (int dim = 0; dim < NUMDIM; dim++) {
#ifdef BINARY
                dis[class_no] += __builtin_popcountll(hv[dim] ^ CLASS[dim][class_no]);
#else
                dis[class_no] += hv[dim] * CLASS[dim][class_no];
#endif
            }
        }
#ifdef BINARY
        int result = std::min_element(dis, dis + NUMCLASS) - dis;
#else
        int result = std::max_element(dis, dis + NUMCLASS) - dis;
#endif
        correct += (result == y[test_no]);
        total++;
        delete[] hv;
    }
    // printf("Accuracy: %f (%d / %d)\n", (float) correct / total, correct, total);
}

// inference with omen
void omen(int &dims, int &correct) {
    // int correct = 0;
    correct = 0;
    int total = 0;
    // int dims = 0;
    dims = 0;
    for (int test_no = 0; test_no < DATA_NUMTEST; test_no++) {
        dis_type dis[NUMCLASS];
        std::memset(dis, 0, sizeof(dis));
#ifdef REAL
        dis_type diff2[NUMCLASS][NUMCLASS];
        std::memset(diff2, 0, sizeof(diff2));
        int diff2_computed[NUMCLASS];
        std::memset(diff2_computed, 0, sizeof(diff2_computed));
        hv_type hv_all[NUMDIM];
#endif
        for (int dim_start = 0, dim_end = FREQ; dim_start < NUMDIM; dim_start = dim_end, dim_end = std::min(dim_start+FREQ, NUMDIM)) { // FREQ should be divided by 64 for binary
            hv_type* hv_chunk = ENCODE_CHUNK(test_no, dim_start, dim_end);
            for (int class_no = 0; class_no < NUMCLASS; class_no++) {
                for (int dim = dim_start; dim < dim_end; dim++) {
                    hv_type hv = hv_chunk[dim - dim_start];
#ifdef BINARY
                    dis[class_no] += __builtin_popcountll(hv ^ CLASS[dim][class_no]);
#else
                    dis[class_no] += hv * CLASS[dim][class_no];
                    hv_all[dim] = hv;
#endif
                }
            }
            delete[] hv_chunk;
            if (dim_end < START) {
                continue;
            }
            // if (dim_end == NUMDIM) {
            //     break;
            // }
#ifdef BINARY
            int cand = std::min_element(dis, dis + NUMCLASS) - dis;
#else
            // calculate diff2
            int cand = std::max_element(dis, dis + NUMCLASS) - dis;
            for (int dim = diff2_computed[cand]; dim < dim_end; dim++) {
                for (int class_no = 0; class_no < NUMCLASS; class_no++) {
                    double tmp = hv_all[dim] * (CLASS[dim][class_no] - CLASS[dim][cand]);
                    diff2[cand][class_no] += tmp * tmp;
                }
            }
            diff2_computed[cand] = dim_end;
#endif  
            // statistical tests with precomputed thresholds
            float W_squared[NUMCLASS];
            for (int class_no = 0; class_no < NUMCLASS; class_no++) {
                if (class_no == cand) {
                    W_squared[class_no] = 1e9;
                } else {
                    double tmp = (dis[class_no]-dis[cand]);
                    tmp *= tmp;
#ifdef BINARY
                    // precomputed diff2 for binary
                    W_squared[class_no] = (tmp * (dim_end<<6)) / (DIFF2[cand][dim_start/FREQ][class_no] * (dim_end<<6) - tmp);
#else
                    W_squared[class_no] = (tmp * dim_end) / (diff2[cand][class_no] * dim_end - tmp);
#endif
                }
            }
            std::sort(W_squared, W_squared + NUMCLASS);
            bool flag = true;
            for (int class_no = 0; class_no < NUMCLASS-1; class_no++) {
                if (W_squared[class_no] < THRESHOLD[class_no]) { // THRESHOLD shape is (NUMCLASS-1)
                    flag = false;
                }
            }
            if (flag || dim_end == NUMDIM) { // pass all tests, return the class with maximum dot product similarity
                correct += (cand == y[test_no]);
                total++;
                dims += dim_end;
                break;
            }
        }
    }
}

#endif

int main() {
    printf("FREQ: %d\n", FREQ);
    printf("START: %d\n", START);
#ifdef BLDC
    printf("FEATURE_HV_DIM: %d\n", FEATURE_HV_DIM);
    printf("VALUE_HV_DIM: %d\n", VALUE_HV_DIM);
#else
    printf("NUMDIM: %d\n", NUMDIM);
#endif
    printf("NUMFEATURE: %d\n", NUMFEATURE);
    printf("NUMCLASS: %d\n", NUMCLASS);
    printf("DATA_NUMTEST: %d\n", DATA_NUMTEST);
    printf("--------------------\n");
    printf("Running normal inference...\n");
    // warmup
    int dims, correct;
    int total = DATA_NUMTEST;
    normal(correct);
    auto start = std::chrono::high_resolution_clock::now();
    normal(correct);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("Time: %f ms\n", elapsed.count());
    printf("Average time per test: %f ms\n", elapsed.count() / total);
    printf("Accuracy: %f (%d / %d)\n", (float) correct / total, correct, total);
    printf("Running omen inference...\n");
    // warmup
    omen(dims, correct);
    start = std::chrono::high_resolution_clock::now();
    omen(dims, correct);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    printf("Time: %f ms\n", elapsed.count());
    printf("Average time per test: %f ms\n", elapsed.count() / total);
    printf("Accuracy: %f (%d / %d)\n", (float) correct / total, correct, total);
    printf("Average dimensions: %f\n", (float) dims / total);
    printf("Done.\n");
    return 0;
}
