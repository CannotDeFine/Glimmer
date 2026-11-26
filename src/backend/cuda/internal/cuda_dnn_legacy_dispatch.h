#pragma once

#include <cudnn.h>

namespace glimmer::backend::cuda {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Old entry points do not exist in SDK9. A defaulted nested record preserves
// existing injected aggregates while keeping every SDK8 ABI compiler-checked.
struct CudaDnnLegacyDispatch {
#if CUDNN_MAJOR == 8
    decltype(&cudnnOpsInferVersionCheck) ops_infer_version{};
    decltype(&cudnnOpsTrainVersionCheck) ops_train_version{};
    decltype(&cudnnCnnInferVersionCheck) cnn_infer_version{};
    decltype(&cudnnCnnTrainVersionCheck) cnn_train_version{};
    decltype(&cudnnAdvInferVersionCheck) adv_infer_version{};
    decltype(&cudnnAdvTrainVersionCheck) adv_train_version{};
    decltype(&cudnnCreateAlgorithmDescriptor) algorithm_create{};
    decltype(&cudnnSetAlgorithmDescriptor) algorithm_set{};
    decltype(&cudnnDestroyAlgorithmDescriptor) algorithm_destroy{};
    decltype(&cudnnSetRNNMatrixMathType) math_set{};
    decltype(&cudnnGetRNNMatrixMathType) math_get{};
    decltype(&cudnnSetRNNBiasMode) bias_set{};
    decltype(&cudnnGetRNNBiasMode) bias_get{};
    decltype(&cudnnSetRNNDescriptor_v6) setup{};
    decltype(&cudnnRNNSetClip) clip{};
    decltype(&cudnnGetRNNWorkspaceSize) workspace_size{};
    decltype(&cudnnGetRNNTrainingReserveSize) reserve_size{};
    decltype(&cudnnGetRNNParamsSize) parameter_size{};
    decltype(&cudnnGetRNNLinLayerMatrixParams) matrix_parameter{};
    decltype(&cudnnGetRNNLinLayerBiasParams) bias_parameter{};
    decltype(&cudnnSetRNNAlgorithmDescriptor) algorithm_bind{};
    decltype(&cudnnGetRNNForwardInferenceAlgorithmMaxCount) inference_max_count{};
    decltype(&cudnnRNNForwardTraining) forward{};
    decltype(&cudnnRNNBackwardData) backward_data{};
    decltype(&cudnnRNNBackwardWeights) backward_weights{};
#endif
};
#pragma GCC diagnostic pop
}  // namespace glimmer::backend::cuda
