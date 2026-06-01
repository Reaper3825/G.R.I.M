#pragma once

#ifdef USE_CUDA

#include "execution_block_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace GRIM::ExecutionBlockInternal {

inline constexpr int kBlockSize = 256;
inline constexpr int kWarpSize = 32;
inline constexpr float kEps = 1e-7f;

inline constexpr int kStageV1 = 1;
inline constexpr int kStageV2 = 2;
inline constexpr int kStageVOut = 3;
inline constexpr int kStageResultEmb = 4;
inline constexpr int kStagePArg1 = 5;
inline constexpr int kStagePArg2 = 6;
inline constexpr int kStagePOp = 7;
inline constexpr int kStagePWrite = 8;

inline constexpr int kStageEntropyArg1 = 21;
inline constexpr int kStageEntropyArg2 = 22;
inline constexpr int kStageEntropyOp = 23;
inline constexpr int kStageWriteCollapse = 24;
inline constexpr int kStageWriteSlotInvalid = 25;
inline constexpr int kStageAtomPosInvalid = 30;

inline constexpr int kStageSlotMissing = 31;
inline constexpr int kStageSlotInvalid = 32;
inline constexpr int kStageSlotUninit = 33;

inline constexpr int kStageTransitionInvalid = 41;
inline constexpr int kStageMultiSlotMutation = 42;

inline const char* stageIdToName(int id) {
	switch (id) {
		case kStageV1: return "v1";
		case kStageV2: return "v2";
		case kStageVOut: return "v_out";
		case kStageResultEmb: return "result_emb";
		case kStagePArg1: return "p_arg1 (softmax validity)";
		case kStagePArg2: return "p_arg2 (softmax validity)";
		case kStagePOp: return "p_op (softmax validity)";
		case kStagePWrite: return "p_write (softmax validity)";
		case kStageEntropyArg1: return "entropy collapse (p_arg1)";
		case kStageEntropyArg2: return "entropy collapse (p_arg2)";
		case kStageEntropyOp: return "entropy collapse (p_op)";
		case kStageWriteCollapse: return "write collapse (max p_write)";
		case kStageWriteSlotInvalid: return "write slot not in value range [S,V)";
		case kStageAtomPosInvalid: return "row-local atom position out of range";
		case kStageSlotMissing: return "missing slot mapping for required state-bearing token";
		case kStageSlotInvalid: return "invalid slot index";
		case kStageSlotUninit: return "slot read before initialization";
		case kStageTransitionInvalid: return "transition error exceeds hard threshold";
		case kStageMultiSlotMutation: return "multiple slots mutated (register machine violation)";
		default: return "unknown";
	}
}

struct StepWorkingSet {
	Tensor cand_hidden;
	Tensor cand_mask;
	Tensor slot_values;
	Tensor context;
	Tensor trace_vec;
	Tensor context_enriched;
	Tensor step_emb;
	// [DELETED] h_arg1, h_arg2 — removed: leaked arg selection into op/write heads.
	Tensor p_arg1;
	Tensor p_arg2;
	Tensor v1;
	Tensor v2;
	Tensor p_op;
	Tensor op_results;
	Tensor v_out;
	Tensor atom_new;
	Tensor v_decoded;
	Tensor result_emb;
	Tensor p_write;
	Tensor state_new;
	Tensor key_new;
	int result_slot = -1;
};

}  // namespace GRIM::ExecutionBlockInternal

#define EXEC_CHECK(cond, msg) \
	do { if (!(cond)) { \
		char buf[512]; \
		snprintf(buf, sizeof(buf), "ExecutionBlock FATAL [%s:%d]: %s", __FILE__, __LINE__, msg); \
		throw std::runtime_error(buf); \
	}} while(0)

#define EXEC_CHECK_SHAPE2(tensor, name, expected_r, expected_c) \
	do { \
		if ((tensor).data == nullptr) { \
			char buf[256]; snprintf(buf, sizeof(buf), "%s: null data pointer", name); \
			EXEC_CHECK(false, buf); \
		} \
		if ((tensor).shape.flat.rows != (expected_r) || (tensor).shape.flat.cols != (expected_c)) { \
			char buf[256]; snprintf(buf, sizeof(buf), \
				"%s: expected [%d, %d], got [%d, %d]", \
				name, (int)(expected_r), (int)(expected_c), \
				(tensor).shape.flat.rows, (tensor).shape.flat.cols); \
			EXEC_CHECK(false, buf); \
		} \
	} while(0)

#define EXEC_CHECK_SHAPE1(tensor, name, expected_n) \
	EXEC_CHECK_SHAPE2(tensor, name, 1, expected_n)

#define CUDA_CHECK(call) \
	do { \
		cudaError_t err__ = (call); \
		if (err__ != cudaSuccess) { \
			char buf[512]; \
			snprintf(buf, sizeof(buf), \
				"ExecutionBlock CUDA FATAL [%s:%d]: %s", \
				__FILE__, __LINE__, cudaGetErrorString(err__)); \
			throw std::runtime_error(buf); \
		} \
	} while(0)

#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

#endif  // USE_CUDA
