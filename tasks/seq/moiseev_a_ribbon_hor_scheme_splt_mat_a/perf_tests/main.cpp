#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/moiseev_a_ribbon_hor_scheme_splt_mat_a/include/ops_seq.hpp"

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_pipeline_run) {
  const size_t m = 100;
  const size_t k = 100;
  const size_t n = 100;

  std::vector<float> A(m * k, 1.0f);
  std::vector<float> B(k * n, 2.0f);
  std::vector<float> C(m * n, 0.0f);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count = {m, k, n};
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskData->outputs_count = {m * n};

  auto testTask =
      std::make_shared<moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<float>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  float expected = m * 2.0f;
  for (size_t i = 0; i < m * n; ++i) {
    ASSERT_EQ(C[i], expected);
  }
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_task_run) {
  const size_t m = 100;
  const size_t k = 100;
  const size_t n = 100;

  std::vector<float> A(m * k, 1.0f);
  std::vector<float> B(k * n, 2.0f);
  std::vector<float> C(m * n, 0.0f);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count = {m, k, n};
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskData->outputs_count = {m * n};

  auto testTask =
      std::make_shared<moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<float>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTask);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);

  float expected = m * 2.0f;
  for (size_t i = 0; i < m * n; ++i) {
    ASSERT_EQ(C[i], expected);
  }
}