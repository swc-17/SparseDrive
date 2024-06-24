#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void deformable_aggregation(
  float* output,
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
);
  

/* feat: bs, num_feat, c */
/* _spatial_shape: cam, scale, 2 */
/* _scale_start_index: cam, scale */
/* _sampling_location: bs, anchor, pts, cam, 2 */
/* _weights: bs, anchor, pts, cam, scale, group */
/* output: bs, anchor, c */
/* kernel: bs, anchor, pts, c */


at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());
  deformable_aggregation(
    output.data_ptr<float>(),
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
  return output;
}


void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
);


void deformable_aggregation_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();
  const float* grad_output = _grad_output.data_ptr<float>();

  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights = _grad_weights.data_ptr<float>();

  deformable_aggregation_grad(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "deformable_aggregation_forward",
    &deformable_aggregation_forward,
    "deformable_aggregation_forward"
  );
  m.def(
    "deformable_aggregation_backward",
    &deformable_aggregation_backward,
    "deformable_aggregation_backward"
  );
}
