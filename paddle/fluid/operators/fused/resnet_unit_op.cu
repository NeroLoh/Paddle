/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_bn_stats_finalize.cu.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_norm_conv.cu.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_scale_bias_add_relu.cu.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class ResNetUnitKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx.GetPlace().GetType() == phi::AllocationType::GPU,
        true,
        phi::errors::PreconditionNotMet("It must use CUDAPlace."));
    PADDLE_ENFORCE_EQ(phi::backends::gpu::CudnnDataType<T>::type,
                      CUDNN_DATA_HALF,
                      phi::errors::Unavailable(
                          "ResNetUnitOp only supports float16 for now."));

    // input x
    const phi::DenseTensor *input_x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor *filter_x = ctx.Input<phi::DenseTensor>("FilterX");
    const phi::DenseTensor *scale_x = ctx.Input<phi::DenseTensor>("ScaleX");
    const phi::DenseTensor *bias_x = ctx.Input<phi::DenseTensor>("BiasX");
    // norm conv
    phi::DenseTensor *conv_out_x = ctx.Output<phi::DenseTensor>("ConvX");
    // bn finalize
    phi::DenseTensor *saved_mean_x = ctx.Output<phi::DenseTensor>("SavedMeanX");
    phi::DenseTensor *saved_invstd_x =
        ctx.Output<phi::DenseTensor>("SavedInvstdX");
    phi::DenseTensor *running_mean_x =
        ctx.Output<phi::DenseTensor>("RunningMeanX");
    phi::DenseTensor *running_var_x =
        ctx.Output<phi::DenseTensor>("RunningVarX");
    // sbar
    phi::DenseTensor *output = ctx.Output<phi::DenseTensor>("Y");
    phi::DenseTensor *bitmask = ctx.Output<phi::DenseTensor>("BitMask");
    // attrs
    int padding = ctx.Attr<int>("padding");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilation = ctx.Attr<int>("dilation");
    int group = ctx.Attr<int>("group");
    double eps = static_cast<double>(ctx.Attr<float>("epsilon"));
    double momentum = static_cast<double>(ctx.Attr<float>("momentum"));
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    bool is_test = ctx.Attr<bool>("is_test");
    bool is_train = !is_test && !use_global_stats;
    std::string act_type = ctx.Attr<std::string>("act_type");

    auto input_x_shape = common::vectorize<int>(input_x->dims());
    auto filter_x_shape = common::vectorize<int>(filter_x->dims());
    // std::swap used to convert shape of filter from conv2d when kernel size is
    // 1.
    if (filter_x_shape[1] != filter_x_shape[2] && 1 == filter_x_shape[2]) {
      std::swap(filter_x_shape[1], filter_x_shape[3]);
    }
    auto param_dims = scale_x->dims();
    auto param_shape = common::vectorize<int>(scale_x->dims());
    if (1 == param_shape.size()) {
      param_shape = {1, 1, 1, param_shape[0]};
    }
    auto output_shape = common::vectorize<int>(output->dims());
    auto bitmask_shape = common::vectorize<int>(bitmask->dims());
    int output_channel = filter_x_shape[0];
    int64_t ele_count = std::accumulate(output_shape.begin(),
                                        output_shape.end(),
                                        1,
                                        std::multiplies<int>()) /
                        output_channel;

    auto place = ctx.GetPlace();
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();

    // 1. Conv
    phi::DenseTensor sum_x;
    phi::DenseTensor sum_of_squares_x;
    sum_x.Resize(param_dims);
    sum_of_squares_x.Resize(param_dims);
    phi::fusion::CudnnNormConvolution<T> conv_x_op(dev_ctx,
                                                   input_x_shape,
                                                   filter_x_shape,
                                                   output_shape,
                                                   padding,
                                                   stride,
                                                   dilation,
                                                   group);
    conv_x_op.Forward(
        dev_ctx, *input_x, *filter_x, conv_out_x, &sum_x, &sum_of_squares_x);

    // 2. BN
    phi::DenseTensor equiv_scale_x;
    phi::DenseTensor equiv_bias_x;
    equiv_scale_x.Resize(param_dims);
    equiv_bias_x.Resize(param_dims);
    phi::fusion::CudnnBNStatsFinalize<T> bn_x_op(dev_ctx, param_shape);
    bn_x_op.Forward(dev_ctx,
                    sum_x,
                    sum_of_squares_x,
                    *scale_x,
                    *bias_x,
                    saved_mean_x,
                    saved_invstd_x,
                    running_mean_x,
                    running_var_x,
                    &equiv_scale_x,
                    &equiv_bias_x,
                    eps,
                    momentum,
                    ele_count,
                    is_train);

    // 3. scale + bias + add + relu
    phi::fusion::CudnnScaleBiasAddRelu<T> sbar_op(dev_ctx,
                                                  act_type,
                                                  fuse_add,
                                                  has_shortcut,
                                                  output_shape,
                                                  param_shape,
                                                  bitmask_shape);
    if (has_shortcut) {
      // input z
      const phi::DenseTensor *input_z = ctx.Input<phi::DenseTensor>("Z");
      const phi::DenseTensor *filter_z = ctx.Input<phi::DenseTensor>("FilterZ");
      const phi::DenseTensor *scale_z = ctx.Input<phi::DenseTensor>("ScaleZ");
      const phi::DenseTensor *bias_z = ctx.Input<phi::DenseTensor>("BiasZ");
      // norm conv
      phi::DenseTensor *conv_out_z = ctx.Output<phi::DenseTensor>("ConvZ");
      // bn finalize
      phi::DenseTensor *saved_mean_z =
          ctx.Output<phi::DenseTensor>("SavedMeanZ");
      phi::DenseTensor *saved_invstd_z =
          ctx.Output<phi::DenseTensor>("SavedInvstdZ");
      phi::DenseTensor *running_mean_z =
          ctx.Output<phi::DenseTensor>("RunningMeanZ");
      phi::DenseTensor *running_var_z =
          ctx.Output<phi::DenseTensor>("RunningVarZ");

      auto input_z_shape = common::vectorize<int>(input_z->dims());
      auto filter_z_shape = common::vectorize<int>(filter_z->dims());

      // 3.1 Conv for second input
      phi::DenseTensor sum_z;
      phi::DenseTensor sum_of_squares_z;
      sum_z.Resize(param_dims);
      sum_of_squares_z.Resize(param_dims);
      phi::fusion::CudnnNormConvolution<T> conv_z_op(dev_ctx,
                                                     input_z_shape,
                                                     filter_z_shape,
                                                     output_shape,
                                                     padding,
                                                     stride_z,
                                                     dilation,
                                                     group);
      conv_z_op.Forward(
          dev_ctx, *input_z, *filter_z, conv_out_z, &sum_z, &sum_of_squares_z);

      // 3.2 BN for second input
      phi::DenseTensor equiv_scale_z;
      phi::DenseTensor equiv_bias_z;
      equiv_scale_z.Resize(param_dims);
      equiv_bias_z.Resize(param_dims);
      phi::fusion::CudnnBNStatsFinalize<T> bn_z_op(dev_ctx, param_shape);
      bn_z_op.Forward(dev_ctx,
                      sum_z,
                      sum_of_squares_z,
                      *scale_z,
                      *bias_z,
                      saved_mean_z,
                      saved_invstd_z,
                      running_mean_z,
                      running_var_z,
                      &equiv_scale_z,
                      &equiv_bias_z,
                      eps,
                      momentum,
                      ele_count,
                      is_train);
      // 3.3 sbar
      sbar_op.Forward(dev_ctx,
                      *conv_out_x,
                      equiv_scale_x,
                      equiv_bias_x,
                      conv_out_z,
                      &equiv_scale_z,
                      &equiv_bias_z,
                      output,
                      bitmask);
    } else {
      const phi::DenseTensor *input_z =
          fuse_add ? ctx.Input<phi::DenseTensor>("Z") : nullptr;
      sbar_op.Forward(dev_ctx,
                      *conv_out_x,
                      equiv_scale_x,
                      equiv_bias_x,
                      input_z,
                      nullptr,
                      nullptr,
                      output,
                      bitmask);
    }
  }
};

template <typename T, typename DeviceContext>
class ResNetUnitGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx.GetPlace().GetType() == phi::AllocationType::GPU,
        true,
        phi::errors::PreconditionNotMet("It must use CUDAPlace."));
    PADDLE_ENFORCE_EQ(phi::backends::gpu::CudnnDataType<T>::type,
                      CUDNN_DATA_HALF,
                      phi::errors::Unavailable(
                          "ResNetUnitOp only supports float16 for now."));

    const phi::DenseTensor *y_grad =
        ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));

    const phi::DenseTensor *x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor *filter_x = ctx.Input<phi::DenseTensor>("FilterX");
    const phi::DenseTensor *scale_x = ctx.Input<phi::DenseTensor>("ScaleX");
    const phi::DenseTensor *bias_x = ctx.Input<phi::DenseTensor>("BiasX");
    const phi::DenseTensor *saved_mean_x =
        ctx.Input<phi::DenseTensor>("SavedMeanX");
    const phi::DenseTensor *saved_invstd_x =
        ctx.Input<phi::DenseTensor>("SavedInvstdX");

    const phi::DenseTensor *conv_out_x = ctx.Input<phi::DenseTensor>("ConvX");
    const phi::DenseTensor *output = ctx.Input<phi::DenseTensor>("Y");
    const phi::DenseTensor *bitmask = ctx.Input<phi::DenseTensor>("BitMask");

    phi::DenseTensor *x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    phi::DenseTensor *filter_x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("FilterX"));
    phi::DenseTensor *scale_x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("ScaleX"));
    phi::DenseTensor *bias_x_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("BiasX"));

    int padding = ctx.Attr<int>("padding");
    int stride = ctx.Attr<int>("stride");
    int stride_z = ctx.Attr<int>("stride_z");
    int dilation = ctx.Attr<int>("dilation");
    int group = ctx.Attr<int>("group");
    double eps = static_cast<double>(ctx.Attr<float>("epsilon"));
    double momentum = static_cast<double>(ctx.Attr<float>("momentum"));
    bool has_shortcut = ctx.Attr<bool>("has_shortcut");
    bool fuse_add = ctx.Attr<bool>("fuse_add");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    std::string act_type = ctx.Attr<std::string>("act_type");

    auto x_shape = common::vectorize<int>(x->dims());
    auto filter_x_shape = common::vectorize<int>(filter_x->dims());
    auto param_shape = common::vectorize<int>(scale_x->dims());
    auto output_shape = common::vectorize<int>(output->dims());
    auto bitmask_shape = common::vectorize<int>(bitmask->dims());

    auto place = ctx.GetPlace();
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();

    // 1. Backward of BN (+ Add + Relu) for x, get conv_out_x_grad,
    // scale_x_grad, bias_x_grad
    phi::DenseTensor conv_out_x_grad;
    conv_out_x_grad.Resize(conv_out_x->dims());
    phi::fusion::CudnnScaleBiasAddRelu<T> sbar_x_op(dev_ctx,
                                                    act_type,
                                                    fuse_add,
                                                    has_shortcut,
                                                    output_shape,
                                                    param_shape,
                                                    bitmask_shape);
    if (has_shortcut) {
      //       X                   Z
      //       |                   |
      //    NormConv            NormConv
      //       |                   |
      // BNStatsFinalize    BNStatsFinalize
      //       \                   /
      //          ScaleBiasAddRelu
      //                  |
      //                  Y
      const phi::DenseTensor *z = ctx.Input<phi::DenseTensor>("Z");
      const phi::DenseTensor *filter_z = ctx.Input<phi::DenseTensor>("FilterZ");
      const phi::DenseTensor *scale_z = ctx.Input<phi::DenseTensor>("ScaleZ");
      const phi::DenseTensor *bias_z = ctx.Input<phi::DenseTensor>("BiasZ");
      const phi::DenseTensor *saved_mean_z =
          ctx.Input<phi::DenseTensor>("SavedMeanZ");
      const phi::DenseTensor *saved_invstd_z =
          ctx.Input<phi::DenseTensor>("SavedInvstdZ");
      const phi::DenseTensor *conv_out_z = ctx.Input<phi::DenseTensor>("ConvZ");

      phi::DenseTensor *z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("Z"));
      phi::DenseTensor *filter_z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("FilterZ"));
      phi::DenseTensor *scale_z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("ScaleZ"));
      phi::DenseTensor *bias_z_grad =
          ctx.Output<phi::DenseTensor>(framework::GradVarName("BiasZ"));

      // 1.1 Backward of BN + Add (+ Relu) for x, get conv_out_x_grad,
      // scale_x_grad, bias_x_grad and z_grad_temp
      phi::DenseTensor z_grad_temp;
      z_grad_temp.Resize(conv_out_z->dims());
      sbar_x_op.Backward(dev_ctx,
                         *y_grad,
                         *conv_out_x,
                         *scale_x,
                         *bias_x,
                         *saved_mean_x,
                         *saved_invstd_x,
                         bitmask,
                         &conv_out_x_grad,
                         &z_grad_temp,
                         scale_x_grad,
                         bias_x_grad,
                         eps);

      // 1.2 bn backward for z, get conv_out_z_grad, dscale_z, dbias_z
      phi::DenseTensor conv_out_z_grad;
      conv_out_z_grad.Resize(conv_out_z->dims());
      phi::fusion::CudnnScaleBiasAddRelu<T> sbar_z_op(
          dev_ctx, "", false, false, output_shape, param_shape, bitmask_shape);
      sbar_z_op.Backward(dev_ctx,
                         z_grad_temp,
                         *conv_out_z,
                         *scale_z,
                         *bias_z,
                         *saved_mean_z,
                         *saved_invstd_z,
                         nullptr,
                         &conv_out_z_grad,
                         nullptr,
                         scale_z_grad,
                         bias_z_grad,
                         eps);

      // 1.3 Backward of Conv for z, get z_grad and filter_z_grad
      auto z_shape = common::vectorize<int>(z->dims());
      auto filter_z_shape = common::vectorize<int>(filter_z->dims());
      phi::fusion::CudnnNormConvolutionGrad<T> conv_z_op(dev_ctx,
                                                         z_shape,
                                                         filter_z_shape,
                                                         output_shape,
                                                         padding,
                                                         stride_z,
                                                         dilation,
                                                         group);
      conv_z_op.Backward(
          dev_ctx, *z, *filter_z, conv_out_z_grad, z_grad, filter_z_grad);
    } else {
      // 1.1 Backward of BN (+ Add + Relu) for x, get conv_out_x_grad,
      // scale_x_grad, bias_x_grad (and z_grad)
      phi::DenseTensor *z_grad =
          fuse_add ? ctx.Output<phi::DenseTensor>(framework::GradVarName("Z"))
                   : nullptr;
      sbar_x_op.Backward(dev_ctx,
                         *y_grad,
                         *conv_out_x,
                         *scale_x,
                         *bias_x,
                         *saved_mean_x,
                         *saved_invstd_x,
                         bitmask,
                         &conv_out_x_grad,
                         z_grad,
                         scale_x_grad,
                         bias_x_grad,
                         eps);
    }

    // 2. Backward of Conv for x, get x_grad and filter_x_grad
    bool use_addto = ctx.Attr<bool>("use_addto");
    phi::fusion::CudnnNormConvolutionGrad<T> conv_x_op(dev_ctx,
                                                       x_shape,
                                                       filter_x_shape,
                                                       output_shape,
                                                       padding,
                                                       stride,
                                                       dilation,
                                                       group);
    conv_x_op.Backward(dev_ctx,
                       *x,
                       *filter_x,
                       conv_out_x_grad,
                       x_grad,
                       filter_x_grad,
                       use_addto);
  }
};

}  // namespace operators
}  // namespace paddle

#if CUDNN_VERSION >= 8000
namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(
    resnet_unit, GPU, ALL_LAYOUT, ops::ResNetUnitKernel, phi::dtype::float16) {}
PD_REGISTER_STRUCT_KERNEL(resnet_unit_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::ResNetUnitGradKernel,
                          phi::dtype::float16) {}
#endif
