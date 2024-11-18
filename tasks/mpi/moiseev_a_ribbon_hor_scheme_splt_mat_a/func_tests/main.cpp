#include <gtest/gtest.h>

#include "mpi/moiseev_a_ribbon_hor_scheme_splt_mat_a/include/ops_mpi.hpp"

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi_test, first_try) {
  using DataType = int32_t;
  size_t m = 2, k = 3, n = 2;
  std::vector<DataType> A = {1, 2, 3, 4, 5, 6};
  std::vector<DataType> B = {7, 8, 9, 10, 11, 12};
  std::vector<DataType> C(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C.size(), m * n);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi::MatrixMultiplicationParallel<DataType> task(taskData);

  //std::cout << "Matrix A:\n";
  //for (size_t i = 0; i < k; ++i) {
  //  for (size_t j = 0; j < n; ++j) {
  //    std::cout << A[i * n + j] << " ";
  //  }
  //  std::cout << "\n";
  //}

  //std::cout << "Matrix B:\n";
  //for (size_t i = 0; i < k; ++i) {
  //  for (size_t j = 0; j < n; ++j) {
  //    std::cout << B[i * n + j] << " ";
  //  }
  //  std::cout << "\n";
  //}
  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<DataType> expected = {58, 64, 139, 154};
  EXPECT_EQ(C, expected);
}