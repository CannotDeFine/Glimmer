#include "driver_frontend_state.h"

#ifdef GLIMMER_HAS_CUDNN
#include "dnn_frontend.inc"
#include "library_info_frontend.inc"
#include "convolution_frontend.inc"
#include "backward_convolution_frontend.inc"
#include "folded_convolution_frontend.inc"
#include "pooling_frontend.inc"
#include "lrn_frontend.inc"
#include "reduction_frontend.inc"
#include "batch_norm_frontend.inc"
#include "dropout_frontend.inc"
#include "op_tensor_frontend.inc"
#include "reorder_filter_frontend.inc"
#include "rnn_data_frontend.inc"
#include "rnn_attention_frontend.inc"
#include "rnn_query_frontend.inc"
#include "rnn_compute_frontend.inc"
#include "rnn_weight_params_frontend.inc"
#if CUDNN_MAJOR == 8
#include "legacy_frontend.inc"
#endif
#include "attn_weights_frontend.inc"
#include "attention_frontend.inc"
#include "backend_graph_frontend.inc"
#endif
