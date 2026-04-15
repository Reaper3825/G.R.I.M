#pragma once

#ifdef USE_CUDA

#include "reasoning_head_GPU.hpp"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace GRIM::ReasoningHeadInternal {

inline constexpr int kBlockSize = 256;

// Stage IDs for device-side error tracking
inline constexpr int kStageGather     = 1;
inline constexpr int kStageConcat     = 2;
inline constexpr int kStageMeanPool   = 3;
inline constexpr int kStageOpLogits   = 4;
inline constexpr int kStageArg1Logits = 5;
inline constexpr int kStageArg2Logits = 6;
inline constexpr int kStageBias       = 7;

inline const char* stageIdToName(int id) {
	switch (id) {
		case kStageGather:     return "gather atom hidden";
		case kStageConcat:     return "concat features";
		case kStageMeanPool:   return "mean pool";
		case kStageOpLogits:   return "op logits projection";
		case kStageArg1Logits: return "arg1 logits projection";
		case kStageArg2Logits: return "arg2 logits projection";
		case kStageBias:       return "op bias addition";
		default:               return "unknown";
	}
}

struct LayerAccess {
	static int* numericErrorFlag(ReasoningHeadLayer& layer) { return layer.d_numeric_error_flag_; }
};

}  // namespace GRIM::ReasoningHeadInternal

#define RH_CHECK(cond, msg) \
	do { if (!(cond)) { \
		char buf[512]; \
		snprintf(buf, sizeof(buf), "ReasoningHead FATAL [%s:%d]: %s", __FILE__, __LINE__, msg); \
		throw std::runtime_error(buf); \
	}} while(0)

#define RH_CHECK_SHAPE2(tensor, name, expected_r, expected_c) \
	do { \
		if ((tensor).data == nullptr) { \
			char buf[256]; snprintf(buf, sizeof(buf), "%s: null data pointer", name); \
			RH_CHECK(false, buf); \
		} \
		if ((tensor).shape.flat.rows != (expected_r) || (tensor).shape.flat.cols != (expected_c)) { \
			char buf[256]; snprintf(buf, sizeof(buf), \
				"%s: expected [%d, %d], got [%d, %d]", \
				name, (int)(expected_r), (int)(expected_c), \
				(tensor).shape.flat.rows, (tensor).shape.flat.cols); \
			RH_CHECK(false, buf); \
		} \
	} while(0)

#define RH_CHECK_SHAPE1(tensor, name, expected_n) \
	RH_CHECK_SHAPE2(tensor, name, 1, expected_n)

#define RH_CUDA_CHECK(call) \
	do { \
		cudaError_t err__ = (call); \
		if (err__ != cudaSuccess) { \
			char buf[512]; \
			snprintf(buf, sizeof(buf), \
				"ReasoningHead CUDA FATAL [%s:%d]: %s", \
				__FILE__, __LINE__, cudaGetErrorString(err__)); \
			throw std::runtime_error(buf); \
		} \
	} while(0)

#define RH_CUDA_CHECK_KERNEL() RH_CUDA_CHECK(cudaGetLastError())

#endif  // USE_CUDA
