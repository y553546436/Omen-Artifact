#include "hdc.h"
#include "model.h"
#include <cstring>
#include <algorithm>
#include <cmath>
#include "timer.h"

#ifdef BINARY
typedef uint64_t hv_type;
typedef int dis_type;
#else // real-valued
typedef double hv_type;
typedef double dis_type;
#endif

/*
 * All capitalized variables and macros are constants defined in hdc.h which is generated from a Python script
 */

#ifdef LANGUAGE
// get the index of a character in the codebook
int get_code_index(int c) {
    return std::lower_bound(CHAR_MAP, CHAR_MAP + NUMCHAR, (int)(c)) - CHAR_MAP;
}
#endif

#ifdef BLDC // binary ldc

float vb_hidden_layer[NUMFEATURE][VALUE_BOX_HIDDEN_DIM];
hv_type vb_output_layer[NUMFEATURE][VALUE_HV_DIM];


void ENCODE_INIT() {
    for (int i = 0; i < NUMFEATURE; i++) {
        for (int j = 0; j < VALUE_BOX_HIDDEN_DIM; j++) {
            vb_hidden_layer[i][j] = TEST_FEATURE[i] * value_box_fc1_weight[j] + value_box_fc1_bias[j];
            // batch normalization
            vb_hidden_layer[i][j] -= vb_bn_running_mean[j];
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
			output += value_box_fc3_bias[dim];
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
hv_type *ENCODE_ALL() {
    ENCODE_INIT();
    hv_type *hv = new hv_type[FEATURE_HV_DIM];
    for (int i = 0; i < FEATURE_HV_DIM; i++) {
        hv[i] = ENCODE(i);
    }
    return hv;
}

// encode chunk of dimensions of a test
void ENCODE_CHUNK(hv_type *hv_chunk, int dim_start, int dim_end) {
    for (int i = dim_start; i < dim_end; i++) {
        hv_chunk[i - dim_start] = ENCODE(i);
    }
}

#else

// encode a test in a specific dimension
hv_type ENCODE(int dim) {
    hv_type hv = 0;
#ifdef REAL
    for (int feature_no = 0; feature_no < NUMFEATURE; feature_no++) {
    	hv += GAUSSIAN[feature_no][dim] * TEST_FEATURE[feature_no];
    }
    hv = std::sin(hv) * std::cos(hv + OFFSET[dim]);
    return hv;
#else
    int cnts[64];
    std::memset(cnts, 0, sizeof(cnts));
#ifdef LANGUAGE
    for (int trigram_start = 1; trigram_start < TEST_FEATURE[0] - 2 + 1; trigram_start++) {
        uint64_t trigram = 0;
        for (int i = 0; i < 3; i++) {
            trigram ^= CHAR_CODEBOOK[get_code_index(TEST_FEATURE[trigram_start + i])][(dim+i)%NUMDIM];
        }
        for (int i = 0; i < 64; i++) {
            cnts[i] += (trigram >> i) & 1;
        }
    }
    for (int i = 0; i < 64; i++) {
        hv |= (hv_type)(cnts[i] > (TEST_FEATURE[0] - 2) / 2) << i;
    }
#else
    for (int feature_no = 0; feature_no < NUMFEATURE; feature_no++) {
        int level = std::ceil((TEST_FEATURE[feature_no] - MIN_VAL) * LEVELS / (MAX_VAL - MIN_VAL));
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
hv_type *ENCODE_ALL() {
    hv_type *hv = new hv_type[NUMDIM];
    for (int dim = 0; dim < NUMDIM; dim++) {
        hv[dim] = ENCODE(dim);
    }
    return hv;
}

// encode chunk of dimensions of a test
void ENCODE_CHUNK(hv_type *hv_chunk, int dim_start, int dim_end) {
    for (int dim = dim_start; dim < dim_end; dim++) {
        hv_chunk[dim - dim_start] = ENCODE(dim);
    }
}

#endif

// normal inference without omen
void normal(uint32_t *result) {
	uint32_t encode_time = 0;
	uint32_t distance_time = 0;
	dis_type dis[NUMCLASS];
	// Encode start
	timer_start();
	hv_type *hv = ENCODE_ALL();
	// Encode stop
	encode_time = timer_stop();
	// Distance start
	timer_start();
	for (int class_no = 0; class_no < NUMCLASS; class_no++) {
		dis[class_no] = 0;
#ifdef BLDC
		for (int dim = 0; dim < FEATURE_HV_DIM; dim++)
#else
		for (int dim = 0; dim < NUMDIM; dim++)
#endif
		{
#ifdef BLDC
			dis[class_no] += hv[dim] ^ class_layer_weight[dim][class_no];
#else
#ifdef BINARY
			dis[class_no] += __builtin_popcountll(hv[dim] ^ CLASS[dim][class_no]);
#else
			dis[class_no] += hv[dim] * CLASS[dim][class_no];
#endif
#endif // BLDC
		}

#ifdef BINARY
        result[0] = std::min_element(dis, dis + NUMCLASS) - dis;
#else
        result[0] = std::max_element(dis, dis + NUMCLASS) - dis;
#endif
	}
	// Distance stop
	distance_time = timer_stop();
	result[1] = encode_time;
	result[2] = distance_time;
	delete[] hv;
}


// inference with omen
void omen(uint32_t *result) {
	uint32_t encode_time = 0;
	uint32_t distance_time = 0;
	uint32_t omen_time = 0;
	dis_type dis[NUMCLASS];
	std::memset(dis, 0, sizeof(dis));
	int dim_start = 0;
	int dim_end = 0;
#ifdef BLDC
	// Encode init start
	timer_start();
	ENCODE_INIT();
	// Encode init stop
	encode_time += timer_stop();
#endif
#ifdef REAL
	// Diff2 init start
	timer_start();
	dis_type diff2[NUMCLASS][NUMCLASS];
	std::memset(diff2, 0, sizeof(diff2));
	int diff2_computed[NUMCLASS];
	std::memset(diff2_computed, 0, sizeof(diff2_computed));
	hv_type hv_all[NUMDIM];
	// Diff2 init stop
	encode_time += timer_stop();
#endif
#ifdef BLDC
	for (dim_start = 0, dim_end = FREQ; dim_start < FEATURE_HV_DIM; dim_start = dim_end, dim_end = std::min(dim_start + FREQ, FEATURE_HV_DIM))
#else
	for (dim_start = 0, dim_end = FREQ; dim_start < NUMDIM; dim_start = dim_end, dim_end = std::min(dim_start+FREQ, NUMDIM)) // FREQ should be divided by 64 for binary
#endif
	{
		// Encode chunk start
		timer_start();
		hv_type hv_chunk[dim_end - dim_start];
		ENCODE_CHUNK(hv_chunk, dim_start, dim_end);
		// Encode chunk stop
		encode_time += timer_stop();
		// Distance start
		timer_start();
		for (int dim = dim_start; dim < dim_end; dim++) {
			hv_type hv = hv_chunk[dim - dim_start];
#ifdef REAL
			hv_all[dim] = hv;
#endif
			for (int class_no = 0; class_no < NUMCLASS; class_no++) {
#ifdef BLDC
				dis[class_no] += hv ^ class_layer_weight[dim][class_no];
#else
#ifdef BINARY
				dis[class_no] += __builtin_popcountll(hv ^ CLASS[dim][class_no]);
#else
				dis[class_no] += hv * CLASS[dim][class_no];
#endif
#endif
			}
		}
		// Distance stop
		distance_time += timer_stop();
		// delete[] hv_chunk;
		if (dim_end < START) {
			continue;
		}
		// Omen start
		timer_start();
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
				break;
			}
		}
#ifdef BLDC
		if (flag || dim_end == FEATURE_HV_DIM)
#else
		if (flag || dim_end == NUMDIM)
#endif
		{ // pass all tests, return the class with maximum dot product similarity
			result[0] = dim_end;
			result[1] = cand;
			break;
		}
		// Omen stop
		omen_time += timer_stop();
    }
	result[2] = encode_time;
	result[3] = distance_time;
	result[4] = omen_time;
}

// inference with absolute strategy
void absolute(uint32_t *result) {
	dis_type dis[NUMCLASS];
	std::memset(dis, 0, sizeof(dis));
	int dim_start = 0;
	int dim_end = 0;
#ifdef REAL
    double hv_squared_sum = 0;
#endif
#ifdef BLDC
	ENCODE_INIT();
#endif
#ifdef BLDC
	for (dim_start = 0, dim_end = FREQ; dim_start < FEATURE_HV_DIM; dim_start = dim_end, dim_end = std::min(dim_start + FREQ, FEATURE_HV_DIM))
#else
	for (dim_start = 0, dim_end = FREQ; dim_start < NUMDIM; dim_start = dim_end, dim_end = std::min(dim_start+FREQ, NUMDIM))
#endif
	{
		hv_type hv_chunk[dim_end - dim_start];
		ENCODE_CHUNK(hv_chunk, dim_start, dim_end);
		for (int dim = dim_start; dim < dim_end; dim++) {
			hv_type hv = hv_chunk[dim - dim_start];
#ifdef REAL
            hv_squared_sum += hv * hv;
#endif
			for (int class_no = 0; class_no < NUMCLASS; class_no++) {
#ifdef BLDC
				dis[class_no] += hv ^ class_layer_weight[dim][class_no];
#else
#ifdef BINARY
				dis[class_no] += __builtin_popcountll(hv ^ CLASS[dim][class_no]);
#else
				dis[class_no] += hv * CLASS[dim][class_no];
#endif
#endif
			}
		}
		if (dim_end < START) {
			continue;
		}
		// calculate cosine similarity
        double similarity[NUMCLASS];
#ifdef BINARY
        for (int class_no = 0; class_no < NUMCLASS; class_no++)
            similarity[class_no] = 1.0 - 1.0 * dis[class_no] / (dim_end<<6);
#else
        for (int class_no = 0; class_no < NUMCLASS; class_no++) {
            similarity[class_no] = dis[class_no] / (std::sqrt(hv_squared_sum * CLASS_SQUARED_SUM[dim_start/FREQ][class_no]));
        }
#endif  
		int cand = std::max_element(similarity, similarity + NUMCLASS) - similarity;
		// check if the largest similarity is larger than threshold
        bool flag = (similarity[cand] > ABSOLUTE_THRESHOLD);
#ifdef BLDC
		if (flag || dim_end == FEATURE_HV_DIM)
#else
		if (flag || dim_end == NUMDIM)
#endif
		{ // pass all tests, return the class with maximum dot product similarity
			result[0] = dim_end;
			result[1] = cand;
			break;
		}
    }
}

// inference with diff strategy
void diff(uint32_t *result) {
	dis_type dis[NUMCLASS];
	std::memset(dis, 0, sizeof(dis));
	int dim_start = 0;
	int dim_end = 0;
#ifdef REAL
    double hv_squared_sum = 0;
#endif
#ifdef BLDC
	ENCODE_INIT();
#endif
#ifdef BLDC
	for (dim_start = 0, dim_end = FREQ; dim_start < FEATURE_HV_DIM; dim_start = dim_end, dim_end = std::min(dim_start + FREQ, FEATURE_HV_DIM))
#else
	for (dim_start = 0, dim_end = FREQ; dim_start < NUMDIM; dim_start = dim_end, dim_end = std::min(dim_start+FREQ, NUMDIM))
#endif
	{
		hv_type hv_chunk[dim_end - dim_start];
		ENCODE_CHUNK(hv_chunk, dim_start, dim_end);
		for (int dim = dim_start; dim < dim_end; dim++) {
			hv_type hv = hv_chunk[dim - dim_start];
#ifdef REAL
            hv_squared_sum += hv * hv;
#endif
			for (int class_no = 0; class_no < NUMCLASS; class_no++) {
#ifdef BLDC
				dis[class_no] += hv ^ class_layer_weight[dim][class_no];
#else
#ifdef BINARY
				dis[class_no] += __builtin_popcountll(hv ^ CLASS[dim][class_no]);
#else
				dis[class_no] += hv * CLASS[dim][class_no];
#endif
#endif
			}
		}
		if (dim_end < START) {
			continue;
		}
		// calculate cosine similarity
        double similarity[NUMCLASS];
#ifdef BINARY
        for (int class_no = 0; class_no < NUMCLASS; class_no++)
            similarity[class_no] = 1.0 - 1.0 * dis[class_no] / (dim_end<<6);
#else
        for (int class_no = 0; class_no < NUMCLASS; class_no++) {
            similarity[class_no] = dis[class_no] / (std::sqrt(hv_squared_sum * CLASS_SQUARED_SUM[dim_start/FREQ][class_no]));
        }
#endif  
		int cand = std::max_element(similarity, similarity + NUMCLASS) - similarity;
		// check if the difference of largest two similarities is larger than threshold
        bool flag = true;
        for (int class_no = 0; class_no < NUMCLASS; class_no++) {
            if (class_no == cand) continue;
            if (similarity[cand] - similarity[class_no] < DIFF_THRESHOLD) {
                flag = false;
                break;
            }
        }
#ifdef BLDC
		if (flag || dim_end == FEATURE_HV_DIM)
#else
		if (flag || dim_end == NUMDIM)
#endif
		{ // pass all tests, return the class with maximum dot product similarity
			result[0] = dim_end;
			result[1] = cand;
			break;
		}
    }
}

// inference with mean strategy
void mean_strategy(uint32_t *result) {
	dis_type dis[NUMCLASS];
	std::memset(dis, 0, sizeof(dis));
	int dim_start = 0;
	int dim_end = 0;
#ifdef REAL
    double hv_squared_sum = 0;
#endif
#ifdef BLDC
	ENCODE_INIT();
#endif
#ifdef BLDC
	for (dim_start = 0, dim_end = FREQ; dim_start < FEATURE_HV_DIM; dim_start = dim_end, dim_end = std::min(dim_start + FREQ, FEATURE_HV_DIM))
#else
	for (dim_start = 0, dim_end = FREQ; dim_start < NUMDIM; dim_start = dim_end, dim_end = std::min(dim_start+FREQ, NUMDIM))
#endif
	{
		hv_type hv_chunk[dim_end - dim_start];
		ENCODE_CHUNK(hv_chunk, dim_start, dim_end);
		for (int dim = dim_start; dim < dim_end; dim++) {
			hv_type hv = hv_chunk[dim - dim_start];
#ifdef REAL
            hv_squared_sum += hv * hv;
#endif
			for (int class_no = 0; class_no < NUMCLASS; class_no++) {
#ifdef BLDC
				dis[class_no] += hv ^ class_layer_weight[dim][class_no];
#else
#ifdef BINARY
				dis[class_no] += __builtin_popcountll(hv ^ CLASS[dim][class_no]);
#else
				dis[class_no] += hv * CLASS[dim][class_no];
#endif
#endif
			}
		}
		if (dim_end < START) {
			continue;
		}
		// calculate cosine similarity
        double similarity[NUMCLASS];
#ifdef BINARY
        for (int class_no = 0; class_no < NUMCLASS; class_no++)
            similarity[class_no] = 1.0 - 1.0 * dis[class_no] / (dim_end<<6);
#else
        for (int class_no = 0; class_no < NUMCLASS; class_no++) {
            similarity[class_no] = dis[class_no] / (std::sqrt(hv_squared_sum * CLASS_SQUARED_SUM[dim_start/FREQ][class_no]));
        }
#endif  
		int cand = std::max_element(similarity, similarity + NUMCLASS) - similarity;
		// check if the smallest distance is smaller than the candidate-dependent threshold derived from training data
        bool flag = (1 - similarity[cand] < MEAN_THRESHOLD[cand]);
#ifdef BLDC
		if (flag || dim_end == FEATURE_HV_DIM)
#else
		if (flag || dim_end == NUMDIM)
#endif
		{ // pass all tests, return the class with maximum dot product similarity
			result[0] = dim_end;
			result[1] = cand;
			break;
		}
    }
}
