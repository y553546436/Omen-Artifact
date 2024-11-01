#include "model.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <chrono>

#ifdef BINARY
typedef uint32_t hv_type;
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
        // printf("Test %d\n", test_no);
        dis_type dis[NUMCLASS];
        hv_type *hv = ENCODE_ALL(test_no);
        // printf("Encoded\n");
        for (int class_no = 0; class_no < NUMCLASS; class_no++) {
            dis[class_no] = 0;
            for (int dim = 0; dim < FEATURE_HV_DIM; dim++) {
#ifdef BINARY
                dis[class_no] += hv[dim] ^ class_layer_weight[dim][class_no];
#else
                dis[class_no] += hv[dim] * CLASS[class_no][dim];
#endif
            }
        }
#ifdef BINARY
        int result = std::min_element(dis, dis + NUMCLASS) - dis;
#else
        int result = std::max_element(dis, dis + NUMCLASS) - dis;
#endif
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

int main() {
    printf("FREQ: %d\n", FREQ);
    printf("START: %d\n", START);
    printf("FEATURE_HV_DIM: %d\n", FEATURE_HV_DIM);
    printf("VALUE_HV_DIM: %d\n", VALUE_HV_DIM);
    printf("NUMFEATURE: %d\n", NUMFEATURE);
    printf("NUMCLASS: %d\n", NUMCLASS);
    printf("NUMTEST: %d\n", DATA_NUMTEST);
    printf("--------------------\n");
    printf("Running normal inference...\n");
    int correct;
    int dims;
    int total = DATA_NUMTEST;
    auto start = std::chrono::high_resolution_clock::now();
    normal(correct);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("Time: %f ms\n", elapsed.count());
    printf("Average time per test: %f ms\n", elapsed.count() / total);
    printf("Accuracy: %f (%d / %d)\n", (float) correct / total, correct, total);
    printf("Running omen inference...\n");
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