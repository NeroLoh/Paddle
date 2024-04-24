# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time

import numpy as np

import paddle
from paddle import inference


class InferenceEngine:
    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.

        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        # (
        #     self.predictor,
        #     self.config,
        #     self.input_tensors,
        #     self.output_tensors,
        # ) = self.load_predictor(
        #     args.model_name + "__model__", args.model_name + "__params__"
        # )
        (
            self.predictor,
            self.config,
            self.input_tensors,
            self.output_tensors,
        ) = self.load_predictor(
            "/home/luowei14/Paddle/build/paddle_inference_install_dir/score_tower/score_tower.pdmodel",
            "/home/luowei14/Paddle/build/paddle_inference_install_dir/score_tower/score_tower.pdiparams",
        )
        # self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
        #      args.model_name, args.model_name
        # )

    def load_predictor(self, model_file_path, params_file_path):
        args = self.args
        # config = inference.Config(model_file_path)
        config = inference.Config(model_file_path, params_file_path)
        # config.enable_lite_engine()
        config.enable_xpu()
        # config.enable_mkldnn()
        xpu_config = paddle.inference.XpuConfig()
        xpu_config.device_id = 0
        # xpu_config.l3_size = 7999488
        # xpu_config.l3_autotune_size = 7999488
        xpu_config.l3_size = 0
        xpu_config.l3_autotune_size = 0
        # set other config from xpu_config
        config.set_xpu_config(xpu_config)
        # config.pass_builder().insert_pass(0, "quant_dequant_xpu_pass")
        # config.delete_pass("memory_optimize_pass")
        # config.delete_pass("constant_folding_pass")
        # config.pass_builder().set_passes([      "quant_dequant_xpu_pass",
        #   "roformer_relative_pos_fuse_pass","multi_encoder_xpu_fuse_pass"])

        config.switch_ir_optim(True)
        config.switch_ir_debug(True)
        config.enable_memory_optim()

        # enable memory optim
        # config.disable_memory_optim()
        # config.disable_glog_info()

        # config.enable_profile()
        # config.switch_use_feed_fetch_ops(False)
        # config.switch_ir_optim(True)

        # config.enable_mkldnn()
        # config.set_cpu_math_library_num_threads(1)
        # config.switch_ir_debug(True)

        # config.exp_disable_tensorrt_ops(["reshape2", "concat", "lookup_table", "lookup_table_v2"])
        # config.delete_pass("trt_embedding_eltwise_layernorm_fuse_pass")
        # config.delete_pass("trt_skip_layernorm_fuse_pass")
        # config.delete_pass("embedding_with_eltwise_add_xpu_fuse_pass")

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        print(input_names)
        input_tensors = []
        output_tensors = []
        for input_name in input_names:
            input_tensor = predictor.get_input_handle(input_name)
            input_tensors.append(input_tensor)
        output_names = predictor.get_output_names()
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
        return predictor, config, input_tensors, output_tensors

    # def preprocess(self, file):
    #     input_keys = ["input_ids", "attention_mask", "bbox", "xpath_name", "xpath_input_ids", "xpath_index", "xpath_mask", "src_feat", "href_feat", "text_need_calc", "text_noneed_calc", "text_noneed_calc_index"]
    #     #data_dict = self.preprocessor.preprocess(file)
    #     result = []
    #     for item in file.split(";"):
    #         shape, vec = item.split(":")
    #         shape = list(map(int, shape.split()))
    #         vec = list(map(int, vec.split()))
    #         result.append(np.array(vec).reshape(shape).astype(np.int64))
    #     return result
    # inputs = [np.array(data_dict[key]).astype(np.int64) for key in input_keys]
    # return inputs

    # def postprocess(self, x):
    #     x = x[0]
    #     x = x.reshape([1, -1, self.args.class_num])
    #     return x

    def run(self, x):
        for idx in range(len(x)):
            print(x[idx].shape)
            print(x[idx].dtype)
            self.input_tensors[idx].reshape(x[idx].shape)
            self.input_tensors[idx].copy_from_cpu(x[idx])
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        return outputs


def get_args(add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="paddle inference", add_help=add_help
    )

    parser.add_argument(
        "--model_name", default="./model/", help="inference model name"
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")

    # parser.add_argument("--data_path", default="fake_input_me.data")
    # parser.add_argument("--class_num", type=int, default=7)

    parser.add_argument("--warmup", default=300, type=int, help="warmup iter")
    parser.add_argument("--iter", default=1000, type=int, help="iter")

    # parser.add_argument(
    #     "--use_tensorrt", default=False, type=str2bool, help="use_tensorrt")
    # parser.add_argument("--precision", default="fp32")
    # parser.add_argument("--min_subgraph_size", type=int, default=10)
    # parser.add_argument("--shape_info_filename", default="trt_shape_info.txt")

    args = parser.parse_args()
    return args


def infer_main(args):
    inference_engine = InferenceEngine(args)
    tmp = np.random.uniform(-0.1, 0.1, (120, 192)).astype("float32")
    print(tmp.dtype)
    # inputs =[tmp,tmp,tmp]
    inputs = []
    inputs.append(tmp)
    dtype = ["int", "int", "int", "float32", "float32", "float32"]
    i = 0
    # with open("test_input1", "r") as f:
    #     lines = f.readlines()
    #     data_inputs = lines[0].split(";")
    #     for item in data_inputs:
    #         array = np.array(
    #             [
    #                 sub_item.strip()
    #                 for sub_item in item.split(":")[1].strip().split(" ")
    #             ]
    #         )
    #         shape = [
    #             int(sub_item.strip())
    #             for sub_item in item.split(":")[0].strip().split(" ")
    #         ]
    #         array = array.reshape(shape)
    #         array = array.astype(dtype[i])
    #         inputs.append(array)
    #         i += 1

    # import paddle.profiler as profiler
    # prof = profiler.Profiler(targets=[profiler.ProfilerTarget.CPU],
    #                scheduler = (0, 100),
    #                on_trace_ready = profiler.export_chrome_tracing('./profiler_log'),
    #                timer_only = False)

    for index in range(args.warmup):
        inference_engine.run(inputs)
        print("warmup_finished", index)
    total_run_cost = []
    # prof.start()
    for index in range(args.iter):
        st = time.time()
        output = inference_engine.run(inputs)
        total_run_cost.append(time.time() - st)
        print(output)
        # prof.step()
    # prof.stop()
    # prof.summary(
    #             op_detail=True,
    #             thread_sep=False,
    #             time_unit='ms')
    print(
        "Average run cost time:",
        str(round(sum(total_run_cost) / len(total_run_cost) * 1000, 3)) + " ms",
    )


if __name__ == "__main__":
    args = get_args()
    infer_main(args)
