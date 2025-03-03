/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#ifdef PADDLE_WITH_HETERPS

#include <google/protobuf/text_format.h>
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef PADDLE_WITH_GLOO
#include <gloo/broadcast.h>

#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif
#include "paddle/fluid/distributed/ps/thirdparty/round_robin.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/fleet/heter_context.h"
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#endif
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps_base.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/heter_util.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/fleet/heter_ps/mem_pool.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#endif
#ifdef PADDLE_WITH_XPU_KP
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#endif
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/platform/place.h"
#ifdef PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/ps/table/accessor.h"
#include "paddle/fluid/distributed/ps/table/ctr_dymf_accessor.h"
#include "paddle/fluid/distributed/ps/wrapper/fleet.h"
#include "paddle/fluid/distributed/the_one_ps.pb.h"
#endif
#ifdef PADDLE_WITH_PSLIB
#include "afs_api.h"            // NOLINT
#include "downpour_accessor.h"  // NOLINT
#endif
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/fleet/heter_ps/log_patch.h"

COMMON_DECLARE_int32(gpugraph_storage_mode);

namespace paddle {
namespace framework {

class Dataset;

#ifdef PADDLE_WITH_PSLIB
class AfsWrapper {
 public:
  AfsWrapper() {}
  virtual ~AfsWrapper() {}
  void init(const std::string& fs_name,
            const std::string& fs_user,
            const std::string& pass_wd,
            const std::string& conf);
  int remove(const std::string& path);
  int mkdir(const std::string& path);
  std::vector<std::string> list(const std::string& path);

  int exist(const std::string& path);
  int upload(const std::string& local_file, const std::string& afs_file);

  int download(const std::string& local_file, const std::string& afs_file);

  int touchz(const std::string& path);
  std::string cat(const std::string& path);
  int mv(const std::string& old_path, const std::string& dest_path);

 private:
  paddle::ps::AfsApiWrapper afs_handler_;
};
#endif

struct task_info {
  std::shared_ptr<char> build_values;
  size_t offset;
  int device_id;
  int multi_mf_dim;
  int start;
  int end;
};

class PSGPUWrapper {
  class DCacheBuffer {
   public:
    DCacheBuffer() : buf_(nullptr) {}
    ~DCacheBuffer() {}
    /**
     * @Brief get data
     */
    template <typename T>
    T* mutable_data(const size_t total_bytes,
                    const paddle::platform::Place& place) {
      if (buf_ == nullptr) {
        buf_ = memory::AllocShared(place, total_bytes);
      } else if (buf_->size() < total_bytes) {
        buf_.reset();
        buf_ = memory::AllocShared(place, total_bytes);
      }
      return reinterpret_cast<T*>(buf_->ptr());
    }
    template <typename T>
    T* data() {
      return reinterpret_cast<T*>(buf_->ptr());
    }
    size_t memory_size() {
      if (buf_ == nullptr) {
        return 0;
      }
      return buf_->size();
    }
    bool IsInitialized(void) { return (buf_ != nullptr); }

   private:
    std::shared_ptr<memory::Allocation> buf_ = nullptr;
  };
  struct PSDeviceData {
    DCacheBuffer keys_tensor;
    DCacheBuffer dims_tensor;
    DCacheBuffer keys_ptr_tensor;
    DCacheBuffer values_ptr_tensor;
    DCacheBuffer pull_push_tensor;

    DCacheBuffer slot_lens;
    DCacheBuffer d_slot_vector;
    DCacheBuffer keys2slot;

    int64_t total_key_length = 0;
    int64_t dedup_key_length = 0;
  };
  PSDeviceData* device_caches_ = nullptr;

 public:
  ~PSGPUWrapper();

  PSGPUWrapper() {
    HeterPs_ = NULL;
    sleep_seconds_before_fail_exit_ = 300;
  }

  void PullSparse(const paddle::platform::Place& place,
                  const int table_id,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const std::vector<int>& slot_dim,
                  const int hidden_size);
  void PullSparse(const paddle::platform::Place& place,
                  const int table_id,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place,
                      const int table_id,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size,
                      const int batch_size);
  void CopyKeys(const paddle::platform::Place& place,
                uint64_t** origin_keys,
                uint64_t* total_keys,
                const int64_t* gpu_len,
                int slot_num,
                int total_len);
  void CopyKeys(const paddle::platform::Place& place,
                uint64_t** origin_keys,
                uint64_t* total_keys,
                const int64_t* gpu_len,
                int slot_num,
                int total_len,
                int* key2slot);

  void divide_to_device(std::shared_ptr<HeterContext> gpu_task);
  void add_slot_feature(std::shared_ptr<HeterContext> gpu_task);
  void BuildGPUTask(std::shared_ptr<HeterContext> gpu_task);
  void PreBuildTask(std::shared_ptr<HeterContext> gpu_task,
                    Dataset* dataset_for_pull);
  void BuildPull(std::shared_ptr<HeterContext> gpu_task);
  void PartitionKey(std::shared_ptr<HeterContext> gpu_task);
  void PrepareGPUTask(std::shared_ptr<HeterContext> gpu_task);
  void LoadIntoMemory(bool is_shuffle);
  void BeginPass();
  void EndPass();
  void add_key_to_local(const std::vector<uint64_t>& keys);
  void add_key_to_gputask(std::shared_ptr<HeterContext> gpu_task);
  void resize_gputask(std::shared_ptr<HeterContext> gpu_task);
  void SparseTableToHbm();
  void HbmToSparseTable();
  void start_build_thread();
  void AddSparseKeys();
  void build_pull_thread();
  void build_task();
  void DumpToMem();
  void MergePull(std::shared_ptr<HeterContext> gpu_task);
  void MergeKeys(std::shared_ptr<HeterContext> gpu_task);
  void FilterPull(std::shared_ptr<HeterContext> gpu_task,
                  const int shard_id,
                  const int dim_id);
  void FilterKey(std::shared_ptr<HeterContext> gpu_task,
                 const int shard_id,
                 const int dim_id);
  // set infer mode
  void SetMode(bool infer_mode) {
    infer_mode_ = infer_mode;
    if (HeterPs_ != NULL) {
      HeterPs_->set_mode(infer_mode);
    }
    VLOG(0) << "set infer mode=" << infer_mode;
  }

  // set sage mode
  void SetSage(bool sage_mode) {
    sage_mode_ = sage_mode;
    VLOG(0) << "set sage mode=" << sage_mode;
  }

  void Finalize() {
    VLOG(3) << "PSGPUWrapper Begin Finalize.";
    if (s_instance_ == nullptr) {
      return;
    }
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
    if (gpu_graph_mode_) {
      if (FLAGS_gpugraph_storage_mode == GpuGraphStorageMode::WHOLE_HBM) {
        this->EndPass();
      }
    }
#endif
    for (size_t i = 0; i < hbm_pools_.size(); i++) {
      delete hbm_pools_[i];
    }
    buildcpu_ready_channel_->Close();
    buildpull_ready_channel_->Close();
    running_ = false;
    VLOG(3) << "begin stop buildpull_threads_";
    buildpull_threads_.join();
    s_instance_ = nullptr;
    VLOG(3) << "PSGPUWrapper Finalize Finished.";
    if (HeterPs_ != NULL) {
      HeterPs_->show_table_collisions();
      delete HeterPs_;
      HeterPs_ = NULL;
    }
    if (device_caches_ != nullptr) {
      delete[] device_caches_;
      device_caches_ = nullptr;
    }
  }

  void InitializeGPU(const std::vector<int>& dev_ids) {
    if (s_instance_ != NULL && is_initialized_ == false) {
      VLOG(3) << "PSGPUWrapper Begin InitializeGPU";
      is_initialized_ = true;
      resource_ = std::make_shared<HeterPsResource>(dev_ids);
      resource_->enable_p2p();
      keys_tensor.resize(resource_->total_device());
      device_caches_ = new PSDeviceData[resource_->total_device()];
#ifdef PADDLE_WITH_GLOO
      auto gloo = paddle::framework::GlooWrapper::GetInstance();
      if (gloo->Size() > 1) {
        multi_node_ = 1;
        resource_->set_multi_node(multi_node_);
        optimizer_config_.multi_node = true;
        VLOG(0) << "init multi node gpu server";
      } else {
        optimizer_config_.multi_node = false;
        VLOG(0) << "init single node gpu server";
      }
#else
      PADDLE_THROW(
          platform::errors::Unavailable("heter ps need compile with GLOO"));
#endif
#ifdef PADDLE_WITH_CUDA
      if (multi_node_) {
        int dev_size = dev_ids.size();
        // init inner comm
        inner_comms_.resize(dev_size);
        inter_ncclids_.resize(dev_size);
        platform::dynload::ncclCommInitAll(
            &(inner_comms_[0]), dev_size, &dev_ids[0]);
// init inter comm
#ifdef PADDLE_WITH_GLOO
        inter_comms_.resize(dev_size);
        if (gloo->Rank() == 0) {
          for (int i = 0; i < dev_size; ++i) {
            platform::dynload::ncclGetUniqueId(&inter_ncclids_[i]);
          }
        }

        PADDLE_ENFORCE_EQ(
            gloo->IsInitialized(),
            true,
            platform::errors::PreconditionNotMet(
                "You must initialize the gloo environment first to use it."));
        gloo::BroadcastOptions opts(gloo->GetContext());
        opts.setOutput(&inter_ncclids_[0], dev_size);
        opts.setRoot(0);
        gloo::broadcast(opts);

        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
        for (int i = 0; i < dev_size; ++i) {
          platform::CUDADeviceGuard guard(dev_ids[i]);
          platform::dynload::ncclCommInitRank(
              &inter_comms_[i], gloo->Size(), inter_ncclids_[i], gloo->Rank());
        }
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());

        rank_id_ = gloo->Rank();
        node_size_ = gloo->Size();
#else
        PADDLE_THROW(
            platform::errors::Unavailable("heter ps need compile with GLOO"));
#endif
      }
#endif
      heter_devices_ = dev_ids;
      buildcpu_ready_channel_->Open();
      buildcpu_ready_channel_->SetCapacity(3);
      buildpull_ready_channel_->Open();
      buildpull_ready_channel_->SetCapacity(1);

      cpu_reday_channels_.resize(dev_ids.size());
      for (size_t i = 0; i < dev_ids.size(); i++) {
        cpu_reday_channels_[i] = paddle::framework::MakeChannel<task_info>();
        cpu_reday_channels_[i]->SetCapacity(16);
      }
      current_task_ = nullptr;

      table_id_ = 0;
      device_num_ = static_cast<int>(heter_devices_.size());

      // start build cpu&gpu ps thread
      start_build_thread();
    }
#ifdef PADDLE_WITH_PSCORE
    cpu_table_accessor_ = fleet_ptr_->worker_ptr_->GetTableAccessor(0);
#endif
#ifdef PADDLE_WITH_PSLIB
    cpu_table_accessor_ =
        fleet_ptr_->pslib_ptr_->_worker_ptr->table_accessor(0);
#endif
    InitializeGPUServer(fleet_ptr_->GetDistDesc());
  }

  void SetSparseSGD(float nonclk_coeff,
                    float clk_coeff,
                    float min_bound,
                    float max_bound,
                    float learning_rate,
                    float initial_g2sum,
                    float initial_range,
                    float beta1_decay_rate,
                    float beta2_decay_rate,
                    float ada_epsilon);
  void SetEmbedxSGD(float mf_create_thresholds,
                    float mf_learning_rate,
                    float mf_initial_g2sum,
                    float mf_initial_range,
                    float mf_min_bound,
                    float mf_max_bound,
                    float mf_beta1_decay_rate,
                    float mf_beta2_decay_rate,
                    float mf_ada_epsilon,
                    float nodeid_slot,
                    float feature_learning_rate);

#ifdef PADDLE_WITH_PSCORE
  void add_sparse_optimizer(
      std::unordered_map<std::string, float>& config,  // NOLINT
      const ::paddle::distributed::SparseCommonSGDRuleParameter& sgd_param,
      const std::string& prefix = "") {
    auto optimizer_name = sgd_param.name();
    if (optimizer_name == "SparseNaiveSGDRule") {
      config[prefix + "optimizer_type"] = 0;
      config[prefix + "learning_rate"] = sgd_param.naive().learning_rate();
      config[prefix + "initial_range"] = sgd_param.naive().initial_range();
      config[prefix + "min_bound"] = sgd_param.naive().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.naive().weight_bounds()[1];
    } else if (optimizer_name == "SparseAdaGradSGDRule") {
      config[prefix + "optimizer_type"] = 1;
      config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
      config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    } else if (optimizer_name == "StdAdaGradSGDRule") {
      config[prefix + "optimizer_type"] = 2;
      config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
      config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    } else if (optimizer_name == "SparseAdamSGDRule") {
      config[prefix + "optimizer_type"] = 3;
      config[prefix + "learning_rate"] = sgd_param.adam().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adam().initial_range();
      config[prefix + "beta1_decay_rate"] = sgd_param.adam().beta1_decay_rate();
      config[prefix + "beta2_decay_rate"] = sgd_param.adam().beta2_decay_rate();
      config[prefix + "ada_epsilon"] = sgd_param.adam().ada_epsilon();
      config[prefix + "min_bound"] = sgd_param.adam().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adam().weight_bounds()[1];
    } else if (optimizer_name == "SparseSharedAdamSGDRule") {
      config[prefix + "optimizer_type"] = 4;
      config[prefix + "learning_rate"] = sgd_param.adam().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adam().initial_range();
      config[prefix + "beta1_decay_rate"] = sgd_param.adam().beta1_decay_rate();
      config[prefix + "beta2_decay_rate"] = sgd_param.adam().beta2_decay_rate();
      config[prefix + "ada_epsilon"] = sgd_param.adam().ada_epsilon();
      config[prefix + "min_bound"] = sgd_param.adam().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adam().weight_bounds()[1];
    } else if (optimizer_name == "SparseAdaGradV2SGDRule") {
      config[prefix + "optimizer_type"] = 5;
      config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
      config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
      config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
      config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
    }
  }

  void InitializeGPUServer(const std::string& dist_desc) {
    paddle::distributed::PSParameter ps_param;
    google::protobuf::TextFormat::ParseFromString(dist_desc, &ps_param);
    auto sparse_table =
        ps_param.server_param().downpour_server_param().downpour_table_param(0);
    // set build thread_num and shard_num
    thread_keys_thread_num_ = sparse_table.shard_num();
    thread_keys_shard_num_ = sparse_table.shard_num();
    VLOG(0) << "ps_gpu build phase thread_num:" << thread_keys_thread_num_
            << " shard_num:" << thread_keys_shard_num_;

    pull_thread_pool_.resize(thread_keys_shard_num_);
    for (size_t i = 0; i < pull_thread_pool_.size(); i++) {
      pull_thread_pool_[i].reset(new ::ThreadPool(1));
    }
    hbm_thread_pool_.resize(device_num_);
    for (size_t i = 0; i < hbm_thread_pool_.size(); i++) {
      hbm_thread_pool_[i].reset(new ::ThreadPool(1));
    }
    cpu_work_pool_.resize(device_num_);
    for (size_t i = 0; i < cpu_work_pool_.size(); i++) {
      cpu_work_pool_[i].reset(new ::ThreadPool(cpu_device_thread_num_));
    }

    auto sparse_table_accessor = sparse_table.accessor();
    auto sparse_table_accessor_parameter =
        sparse_table_accessor.ctr_accessor_param();
    accessor_class_ = sparse_table_accessor.accessor_class();

    std::unordered_map<std::string, float> config;
    config["embedx_dim"] = sparse_table_accessor.embedx_dim();
    config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
    config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
    config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold();

    config["nodeid_slot"] =
        sparse_table_accessor.graph_sgd_param().nodeid_slot();
    config["feature_learning_rate"] =
        sparse_table_accessor.graph_sgd_param().feature_learning_rate();

    if (accessor_class_ == "CtrDymfAccessor") {
      // optimizer config for embed_w and embedx
      add_sparse_optimizer(config, sparse_table_accessor.embed_sgd_param());
      add_sparse_optimizer(
          config, sparse_table_accessor.embedx_sgd_param(), "mf_");
    }

    fleet_config_ = config;
    GlobalAccessorFactory::GetInstance().Init(accessor_class_);
    GlobalAccessorFactory::GetInstance().GetAccessorWrapper()->Configure(
        config);
    InitializeGPUServer(config);
  }
#endif

#ifdef PADDLE_WITH_PSLIB
  void add_sparse_optimizer(
      std::unordered_map<std::string, float>& config,  // NOLINT
      const paddle::SparseCommonSGDRuleParameter& sgd_param,
      const std::string& prefix = "") {
    auto optimizer_name = sgd_param.name();
    if (optimizer_name == "naive") {
      config[prefix + "optimizer_type"] = 0;
      config[prefix + "learning_rate"] = sgd_param.naive().learning_rate();
      config[prefix + "initial_range"] = sgd_param.naive().initial_range();
      if (sgd_param.naive().weight_bounds_size() == 2) {
        config[prefix + "min_bound"] = sgd_param.naive().weight_bounds()[0];
        config[prefix + "max_bound"] = sgd_param.naive().weight_bounds()[1];
      }
    } else if (optimizer_name == "adagrad") {
      config[prefix + "optimizer_type"] = 1;
      config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
      config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
      if (sgd_param.adagrad().weight_bounds_size() == 2) {
        config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
        config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
      }
    } else if (optimizer_name == "std_adagrad") {
      config[prefix + "optimizer_type"] = 2;
      config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
      config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
      if (sgd_param.adagrad().weight_bounds_size() == 2) {
        config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
        config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
      }
    } else if (optimizer_name == "adam") {
      config[prefix + "optimizer_type"] = 3;
      config[prefix + "learning_rate"] = sgd_param.adam().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adam().initial_range();
      if (sgd_param.adam().weight_bounds_size() == 2) {
        config[prefix + "min_bound"] = sgd_param.adam().weight_bounds()[0];
        config[prefix + "max_bound"] = sgd_param.adam().weight_bounds()[1];
      }
    } else if (optimizer_name == "adagrad_v2") {
      config[prefix + "optimizer_type"] = 5;
      config[prefix + "learning_rate"] = sgd_param.adagrad().learning_rate();
      config[prefix + "initial_range"] = sgd_param.adagrad().initial_range();
      config[prefix + "initial_g2sum"] = sgd_param.adagrad().initial_g2sum();
      if (sgd_param.adagrad().weight_bounds_size() == 2) {
        config[prefix + "min_bound"] = sgd_param.adagrad().weight_bounds()[0];
        config[prefix + "max_bound"] = sgd_param.adagrad().weight_bounds()[1];
      }
    }
  }

  void InitializeGPUServer(const std::string& dist_desc) {
    // optimizer config for hbmps
    paddle::PSParameter ps_param;
    google::protobuf::TextFormat::ParseFromString(dist_desc, &ps_param);
    auto sparse_table =
        ps_param.server_param().downpour_server_param().downpour_table_param(0);
    auto sparse_table_accessor = sparse_table.accessor();
    auto sparse_table_accessor_parameter =
        sparse_table_accessor.downpour_accessor_param();
    accessor_class_ = sparse_table_accessor.accessor_class();

    // NOTE(zhangminxu): gpups' sparse table optimizer config,
    // now only support single sparse table
    // auto sparse_table = param_.sparse_table(0);
    std::unordered_map<std::string, float> config;
    if (accessor_class_ == "DownpourFeatureValueAccessor" ||
        accessor_class_ == "DownpourCtrAccessor" ||
        accessor_class_ == "DownpourCtrDoubleAccessor") {
      config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
      config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
      config["learning_rate"] =
          sparse_table_accessor.sparse_sgd_param().learning_rate();
      config["initial_g2sum"] =
          sparse_table_accessor.sparse_sgd_param().initial_g2sum();
      config["initial_range"] =
          sparse_table_accessor.sparse_sgd_param().initial_range();
      if (sparse_table_accessor.sparse_sgd_param().weight_bounds_size() == 2) {
        config["min_bound"] =
            sparse_table_accessor.sparse_sgd_param().weight_bounds()[0];
        config["max_bound"] =
            sparse_table_accessor.sparse_sgd_param().weight_bounds()[1];
      }
      // NOTE(zhangminxu): for DownpourCtrAccessor & DownpourCtrDoubleAccessor,
      // optimizer config for embed_w & embedx_w is the same
      config["mf_create_thresholds"] =
          sparse_table_accessor.embedx_threshold();  // default = 10
      config["mf_learning_rate"] = config["learning_rate"];
      config["mf_initial_g2sum"] = config["initial_g2sum"];
      config["mf_initial_range"] = config["initial_range"];
      config["mf_min_bound"] = config["min_bound"];
      config["mf_max_bound"] = config["max_bound"];
      config["mf_embedx_dim"] =
          sparse_table_accessor.embedx_dim();  // default = 8

    } else if (accessor_class_ == "DownpourSparseValueAccessor") {
      auto optimizer_name =
          sparse_table_accessor.sparse_commonsgd_param().name();
      if (optimizer_name == "naive") {
        config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                      .naive()
                                      .learning_rate();
        config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                      .naive()
                                      .initial_range();
        if (sparse_table_accessor.sparse_commonsgd_param()
                .naive()
                .weight_bounds_size() == 2) {
          config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .weight_bounds()[0];
          config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .naive()
                                    .weight_bounds()[1];
        }
      } else if (optimizer_name == "adagrad") {
        config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                      .adagrad()
                                      .learning_rate();
        config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                      .adagrad()
                                      .initial_range();
        config["initial_g2sum"] = sparse_table_accessor.sparse_commonsgd_param()
                                      .adagrad()
                                      .initial_g2sum();
        if (sparse_table_accessor.sparse_commonsgd_param()
                .adagrad()
                .weight_bounds_size() == 2) {
          config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .weight_bounds()[0];
          config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adagrad()
                                    .weight_bounds()[1];
        }
      } else if (optimizer_name == "adam") {
        config["learning_rate"] = sparse_table_accessor.sparse_commonsgd_param()
                                      .adam()
                                      .learning_rate();
        config["initial_range"] = sparse_table_accessor.sparse_commonsgd_param()
                                      .adam()
                                      .initial_range();
        if (sparse_table_accessor.sparse_commonsgd_param()
                .adam()
                .weight_bounds_size() == 2) {
          config["min_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adam()
                                    .weight_bounds()[0];
          config["max_bound"] = sparse_table_accessor.sparse_commonsgd_param()
                                    .adam()
                                    .weight_bounds()[1];
        }
      }
    } else if (accessor_class_ == "DownpourUnitAccessor" ||
               accessor_class_ == "DownpourDoubleUnitAccessor" ||
               accessor_class_ == "DownpourCtrDymfAccessor" ||
               accessor_class_ == "DownpourCtrDoubleDymfAccessor") {
      config["nonclk_coeff"] = sparse_table_accessor_parameter.nonclk_coeff();
      config["clk_coeff"] = sparse_table_accessor_parameter.click_coeff();
      config["mf_create_thresholds"] = sparse_table_accessor.embedx_threshold();
      // optimizer config for embed_w and embedx
      add_sparse_optimizer(config, sparse_table_accessor.embed_sgd_param());
      add_sparse_optimizer(
          config, sparse_table_accessor.embedx_sgd_param(), "mf_");
      config["mf_embedx_dim"] =
          sparse_table_accessor.embedx_dim();  // default = 8
    }
    config["sparse_shard_num"] = sparse_table.shard_num();

    GlobalAccessorFactory::GetInstance().Init(accessor_class_);

    GlobalAccessorFactory::GetInstance().GetAccessorWrapper()->Configure(
        config);

    InitializeGPUServer(config);
  }
#endif

  void InitializeGPUServer(std::unordered_map<std::string, float> config) {
    float nonclk_coeff = (config.find("nonclk_coeff") == config.end())
                             ? 1.0
                             : config["nonclk_coeff"];
    float clk_coeff =
        (config.find("clk_coeff") == config.end()) ? 1.0 : config["clk_coeff"];
    float min_bound = (config.find("min_bound") == config.end())
                          ? -10.0
                          : config["min_bound"];
    float max_bound =
        (config.find("max_bound") == config.end()) ? 10.0 : config["max_bound"];
    float learning_rate = (config.find("learning_rate") == config.end())
                              ? 0.05
                              : config["learning_rate"];
    float initial_g2sum = (config.find("initial_g2sum") == config.end())
                              ? 3.0
                              : config["initial_g2sum"];
    float initial_range = (config.find("initial_range") == config.end())
                              ? 1e-4
                              : config["initial_range"];
    float beta1_decay_rate = (config.find("beta1_decay_rate") == config.end())
                                 ? 0.9
                                 : config["beta1_decay_rate"];
    float beta2_decay_rate = (config.find("beta2_decay_rate") == config.end())
                                 ? 0.999
                                 : config["beta2_decay_rate"];
    float ada_epsilon = (config.find("ada_epsilon") == config.end())
                            ? 1e-8
                            : config["ada_epsilon"];
    // mf config settings
    float mf_create_thresholds =
        (config.find("mf_create_thresholds") == config.end())
            ? static_cast<float>(1.0)
            : config["mf_create_thresholds"];
    float mf_learning_rate = (config.find("mf_learning_rate") == config.end())
                                 ? 0.05
                                 : config["mf_learning_rate"];
    float mf_initial_g2sum = (config.find("mf_initial_g2sum") == config.end())
                                 ? 3.0
                                 : config["mf_initial_g2sum"];
    float mf_initial_range = (config.find("mf_initial_range") == config.end())
                                 ? 1e-4
                                 : config["mf_initial_range"];
    float mf_min_bound = (config.find("mf_min_bound") == config.end())
                             ? -10.0
                             : config["mf_min_bound"];
    float mf_max_bound = (config.find("mf_max_bound") == config.end())
                             ? 10.0
                             : config["mf_max_bound"];
    float mf_beta1_decay_rate =
        (config.find("mf_beta1_decay_rate") == config.end())
            ? 0.9
            : config["mf_beta1_decay_rate"];
    float mf_beta2_decay_rate =
        (config.find("mf_beta2_decay_rate") == config.end())
            ? 0.999
            : config["mf_beta2_decay_rate"];
    float mf_ada_epsilon = (config.find("mf_ada_epsilon") == config.end())
                               ? 1e-8
                               : config["mf_ada_epsilon"];

    float feature_learning_rate =
        (config.find("feature_learning_rate") == config.end())
            ? 0.05
            : config["feature_learning_rate"];

    float nodeid_slot = (config.find("nodeid_slot") == config.end())
                            ? 9008
                            : config["nodeid_slot"];

    this->SetSparseSGD(nonclk_coeff,
                       clk_coeff,
                       min_bound,
                       max_bound,
                       learning_rate,
                       initial_g2sum,
                       initial_range,
                       beta1_decay_rate,
                       beta2_decay_rate,
                       ada_epsilon);
    this->SetEmbedxSGD(mf_create_thresholds,
                       mf_learning_rate,
                       mf_initial_g2sum,
                       mf_initial_range,
                       mf_min_bound,
                       mf_max_bound,
                       mf_beta1_decay_rate,
                       mf_beta2_decay_rate,
                       mf_ada_epsilon,
                       nodeid_slot,
                       feature_learning_rate);

    // set optimizer type(naive,adagrad,std_adagrad,adam,share_adam)
    optimizer_type_ = (config.find("optimizer_type") == config.end())
                          ? 1
                          : static_cast<int>(config["optimizer_type"]);

    VLOG(0) << "InitializeGPUServer optimizer_type_:" << optimizer_type_
            << " nodeid_slot:" << nodeid_slot
            << " feature_learning_rate:" << feature_learning_rate;
  }

  void SetDate(int year, int month, int day) {
    year_ = year;
    month_ = month;
    day_ = day;
  }

  void SetDataset(Dataset* dataset) { dataset_ = dataset; }

  // PSGPUWrapper singleton
  static std::shared_ptr<PSGPUWrapper> GetInstance() {
    {
      std::lock_guard<std::mutex> lk(ins_mutex);
      if (NULL == s_instance_) {
        s_instance_.reset(new paddle::framework::PSGPUWrapper());
      }
    }
    return s_instance_;
  }
  std::vector<std::unordered_map<uint64_t, std::vector<float>>>& GetLocalTable(
      int table_id) {
    return local_tables_[table_id];
  }
  void SetSlotVector(const std::vector<int>& slot_vector) {
    slot_vector_ = slot_vector;
    VLOG(0) << "slot_vector size is " << slot_vector_.size();
  }
  void SetPullFeatureSlotNum(int sparse_slot_num, int float_slot_num) {
    slot_num_for_pull_feature_ = sparse_slot_num;
    float_slot_num_ = float_slot_num;
#if defined(PADDLE_WITH_PSCORE) && defined(PADDLE_WITH_HETERPS)
    if (gpu_graph_mode_) {
      auto gpu_graph_ptr = GraphGpuWrapper::GetInstance();
      gpu_graph_ptr->set_feature_info(slot_num_for_pull_feature_,
                                      float_slot_num_);
    }
#endif
    VLOG(0) << "slot_num_for_pull_feature_ is " << slot_num_for_pull_feature_
            << ", float_slot_num is " << float_slot_num_;
  }
  void SetSlotOffsetVector(const std::vector<int>& slot_offset_vector) {
    slot_offset_vector_ = slot_offset_vector;
    std::cout << "yxf set: ";
    for (auto s : slot_offset_vector_) {
      std::cout << s << " | ";
    }
    std::cout << " end " << std::endl;
  }

#ifdef PADDLE_WITH_CUDA
  void SetSlotDimVector(const std::vector<int>& slot_mf_dim_vector) {
    slot_mf_dim_vector_ = slot_mf_dim_vector;
    assert(slot_mf_dim_vector_.size() == slot_vector_.size());
  }

  void InitSlotInfo() {
    if (slot_info_initialized_) {
      return;
    }
    SlotRecordDataset* dataset = reinterpret_cast<SlotRecordDataset*>(dataset_);
    auto slots_vec = dataset->GetSlots();
    slot_offset_vector_.clear();
    for (auto& slot : slot_vector_) {
      for (size_t i = 0; i < slots_vec.size(); ++i) {
        if (std::to_string(slot) == slots_vec[i]) {
          slot_offset_vector_.push_back(i);
          break;
        }
      }
    }
    std::cout << "psgpu wrapper use slots: ";
    for (auto s : slot_offset_vector_) {
      std::cout << s << " | ";
    }
    std::cout << " end " << std::endl;
    for (size_t i = 0; i < slot_mf_dim_vector_.size(); i++) {
      slot_dim_map_[slot_vector_[i]] = slot_mf_dim_vector_[i];
    }

    std::unordered_set<int> dims_set;
    for (auto& it : slot_dim_map_) {
      dims_set.insert(it.second);
    }
    size_t num_of_dim = dims_set.size();
    index_dim_vec_.resize(num_of_dim);
    index_dim_vec_.assign(dims_set.begin(), dims_set.end());
    std::sort(index_dim_vec_.begin(), index_dim_vec_.end());
    std::unordered_map<int, int> dim_index_map;
    for (size_t i = 0; i < num_of_dim; i++) {
      dim_index_map[index_dim_vec_[i]] = i;
    }
    hbm_pools_.resize(resource_->total_device() * num_of_dim);
    for (size_t i = 0; i < hbm_pools_.size(); i++) {
      hbm_pools_[i] = new HBMMemoryPoolFix();
    }

    mem_pools_.resize(resource_->total_device() * num_of_dim);
    max_mf_dim_ = index_dim_vec_.back();
    multi_mf_dim_ = (dim_index_map.size() >= 1) ? dim_index_map.size() : 0;
    resource_->set_multi_mf(multi_mf_dim_, max_mf_dim_);
    slot_index_vec_.resize(slot_mf_dim_vector_.size());
    for (size_t i = 0; i < slot_index_vec_.size(); i++) {
      slot_index_vec_[i] = dim_index_map[slot_mf_dim_vector_[i]];
    }

    auto accessor_wrapper_ptr =
        GlobalAccessorFactory::GetInstance().GetAccessorWrapper();
    val_type_size_ = accessor_wrapper_ptr->GetFeatureValueSize(max_mf_dim_);
    grad_type_size_ = accessor_wrapper_ptr->GetPushValueSize(max_mf_dim_);
    pull_type_size_ = accessor_wrapper_ptr->GetPullValueSize(max_mf_dim_);
    VLOG(0) << "InitSlotInfo: val_type_size_" << val_type_size_
            << " grad_type_size_:" << grad_type_size_
            << " pull_type_size_:" << pull_type_size_;
    slot_info_initialized_ = true;
  }
#endif

  void ShowOneTable(int index) { HeterPs_->show_one_table(index); }

  int UseAfsApi() { return use_afs_api_; }

#ifdef PADDLE_WITH_PSLIB
  std::shared_ptr<paddle::ps::AfsReader> OpenReader(
      const std::string& filename) {
    return afs_handler_.open_reader(filename);
  }

  std::shared_ptr<paddle::ps::AfsWriter> OpenWriter(
      const std::string& filename) {
    return afs_handler_.open_writer(filename);
  }

  void InitAfsApi(const std::string& fs_name,
                  const std::string& fs_user,
                  const std::string& pass_wd,
                  const std::string& conf);
#endif

  // for node rank
  int PartitionKeyForRank(const uint64_t& key) {
    return static_cast<int>((key / device_num_) % node_size_);
  }
  // is key for self rank
  bool IsKeyForSelfRank(const uint64_t& key) {
    return (static_cast<int>((key / device_num_) % node_size_) == rank_id_);
  }
  // rank id
  int GetRankId(void) { return rank_id_; }
  // rank size
  int GetRankNum(void) { return node_size_; }
  // rank id
  int GetNCCLRankId(const int& device_id) {
    return (rank_id_ * device_num_ + device_id);
  }

 private:
  static std::shared_ptr<PSGPUWrapper> s_instance_;
  static std::mutex ins_mutex;
  Dataset* dataset_;
#ifdef PADDLE_WITH_PSLIB
  paddle::ps::AfsApiWrapper afs_handler_;
#endif
  std::unordered_map<
      uint64_t,
      std::vector<std::unordered_map<uint64_t, std::vector<float>>>>
      local_tables_;
  HeterPsBase* HeterPs_ = NULL;
  // std::vector<LoDTensor> keys_tensor;  // Cache for pull_sparse
  std::vector<phi::DenseTensor> keys_tensor;  // Cache for pull_sparse
  std::shared_ptr<HeterPsResource> resource_;
  int32_t sleep_seconds_before_fail_exit_;
  std::vector<int> slot_vector_;
  std::vector<int> slot_offset_vector_;
  std::vector<int> slot_mf_dim_vector_;
  std::unordered_map<int, int> slot_dim_map_;
  std::vector<int> slot_index_vec_;
  std::vector<int> index_dim_vec_;
  int multi_mf_dim_{0};
  int max_mf_dim_{0};
  int slot_num_for_pull_feature_{0};
  int float_slot_num_{0};
  size_t val_type_size_{0};
  size_t grad_type_size_{0};
  size_t pull_type_size_{0};

  double time_1 = 0.0;
  double time_2 = 0.0;
  double time_3 = 0.0;
  double time_4 = 0.0;

  int multi_node_{0};
  int rank_id_ = 0;
  int node_size_ = 1;
  int device_num_ = 8;
  uint64_t table_id_;
  int gpu_graph_mode_ = 0;
#ifdef PADDLE_WITH_CUDA
  std::vector<ncclComm_t> inner_comms_;
  std::vector<ncclComm_t> inter_comms_;
  std::vector<ncclUniqueId> inter_ncclids_;
#endif
  std::vector<int> heter_devices_;
  std::unordered_set<std::string> gpu_ps_config_keys_;
  HeterObjectPool<HeterContext> gpu_task_pool_;
  std::vector<std::vector<robin_hood::unordered_set<uint64_t>>> thread_keys_;
  std::vector<std::vector<std::vector<robin_hood::unordered_set<uint64_t>>>>
      thread_dim_keys_;
  int thread_keys_thread_num_ = 37;
  int thread_keys_shard_num_ = 37;
  uint64_t max_fea_num_per_pass_ = 5000000000;
  int year_;
  int month_;
  int day_;
  bool slot_info_initialized_ = false;
  bool hbm_sparse_table_initialized_ = false;
  int use_afs_api_ = 0;
  int optimizer_type_ = 1;
  std::string accessor_class_;
  std::unordered_map<std::string, float> fleet_config_;
#ifdef PADDLE_WITH_PSCORE
  std::shared_ptr<paddle::distributed::FleetWrapper> fleet_ptr_ =
      paddle::distributed::FleetWrapper::GetInstance();
  paddle::distributed::ValueAccessor* cpu_table_accessor_;
#endif

#ifdef PADDLE_WITH_PSLIB
  std::shared_ptr<FleetWrapper> fleet_ptr_ = FleetWrapper::GetInstance();
  paddle::ps::ValueAccessor* cpu_table_accessor_;
#endif

#ifdef PADDLE_WITH_CUDA
  std::vector<MemoryPool*> mem_pools_;
  std::vector<HBMMemoryPoolFix*> hbm_pools_;  // in multi mfdim, one table need
                                              // hbm pools of total dims number
#endif

  std::shared_ptr<
      paddle::framework::ChannelObject<std::shared_ptr<HeterContext>>>
      buildcpu_ready_channel_ =
          paddle::framework::MakeChannel<std::shared_ptr<HeterContext>>();
  std::shared_ptr<
      paddle::framework::ChannelObject<std::shared_ptr<HeterContext>>>
      buildpull_ready_channel_ =
          paddle::framework::MakeChannel<std::shared_ptr<HeterContext>>();
  std::vector<std::shared_ptr<paddle::framework::ChannelObject<task_info>>>
      cpu_reday_channels_;
  std::shared_ptr<HeterContext> current_task_ = nullptr;
  std::thread buildpull_threads_;
  bool running_ = false;
  std::vector<std::shared_ptr<::ThreadPool>> pull_thread_pool_;
  std::vector<std::shared_ptr<::ThreadPool>> hbm_thread_pool_;
  std::vector<std::shared_ptr<::ThreadPool>> cpu_work_pool_;
  OptimizerConfig optimizer_config_;
  // gradient push count
  uint64_t grad_push_count_ = 0;
  // infer mode
  bool infer_mode_ = false;
  // sage mode
  bool sage_mode_ = false;
  size_t cpu_device_thread_num_ = 16;

 protected:
  static bool is_initialized_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
