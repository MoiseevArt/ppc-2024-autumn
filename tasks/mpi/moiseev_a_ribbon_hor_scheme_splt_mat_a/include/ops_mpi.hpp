#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>

#include "core/task/include/task.hpp"

namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi {

template <typename DataType>
class MatrixMultiplicationParallel : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), taskData(taskData_) {}

  bool pre_processing() override {
    internal_order_test();

    m = taskData->inputs_count[0];
    k = taskData->inputs_count[1];
    n = taskData->inputs_count[2];

    auto tmp_ptr_A = reinterpret_cast<DataType*>(taskData->inputs[0]);
    auto tmp_ptr_B = reinterpret_cast<DataType*>(taskData->inputs[1]);
    A.assign(tmp_ptr_A, tmp_ptr_A + m * k);
    B.assign(tmp_ptr_B, tmp_ptr_B + k * n);

    if (A.size() != m * k || B.size() != k * n) {
      return false;
    }

    C.resize(m * n);
    std::fill(C.begin(), C.end(), 0);
    return true;
  }

  bool validation() override {
    internal_order_test();
    return (world.rank() != 0 ||
            (taskData->inputs.size() == 2 && taskData->inputs_count.size() == 3 && m * k == n * k));
  }

  bool run() override {
    internal_order_test();

    size_t rank = static_cast<size_t>(world.rank());
    size_t size = static_cast<size_t>(world.size());

    size_t rows_per_proc = m / size;
    size_t remainder = m % size;

    size_t start_row = rank * rows_per_proc + std::min(rank, remainder);
    size_t end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    size_t local_row_count = end_row - start_row;

    boost::mpi::broadcast(world, local_row_count, 0);
    boost::mpi::broadcast(world, rows_per_proc, 0);
    boost::mpi::broadcast(world, remainder, 0);

    std::vector<DataType> local_A(local_row_count * k);

    if (rank == 0) {
      std::vector<int> sendcounts(size), displs(size);
      for (size_t i = 0; i < size; ++i) {
        size_t proc_start_row = i * rows_per_proc + std::min(i, remainder);
        size_t proc_row_count = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = static_cast<int>(proc_row_count * k);
        displs[i] = static_cast<int>(proc_start_row * k);
      }
      boost::mpi::scatterv(world, A.data(), sendcounts, displs, local_A.data(), local_A.size(), 0);
    } else {
      boost::mpi::scatterv(world, static_cast<DataType*>(nullptr), {}, {}, local_A.data(), local_A.size(), 0);
    }
    boost::mpi::broadcast(world, B, 0);

    std::vector<DataType> local_C(local_row_count * n, 0);

    for (size_t i = 0; i < local_row_count; ++i) {
      for (size_t j = 0; j < n; ++j) {
        for (size_t p = 0; p < k; ++p) {
          local_C[i * n + j] += local_A[i * k + p] * B[p * n + j];
        }
      }
    }

    if (rank == 0) {
      std::vector<int> recvcounts(size), displs(size);
      for (size_t i = 0; i < size; ++i) {
        size_t proc_start_row = i * rows_per_proc + std::min(i, remainder);
        size_t proc_row_count = rows_per_proc + (i < remainder ? 1 : 0);
        recvcounts[i] = static_cast<int>(proc_row_count * n);
        displs[i] = static_cast<int>(proc_start_row * n);
      }
      boost::mpi::gatherv(world, local_C.data(), local_C.size(), C.data(), recvcounts, displs, 0);
    } else {
      boost::mpi::gatherv(world, local_C.data(), local_C.size(), C.data(), {}, {}, 0);
    }

    return true;
  }

  bool post_processing() override {
    internal_order_test();

    if (taskData->outputs.size() >= 1) {
      auto output_ptr = reinterpret_cast<DataType*>(taskData->outputs[0]);
      std::copy(C.begin(), C.end(), output_ptr);
      return true;
    }
    return false;
  }

 private:
  std::shared_ptr<ppc::core::TaskData> taskData;
  boost::mpi::communicator world;
  std::vector<DataType> A;
  std::vector<DataType> B;
  std::vector<DataType> C;
  size_t m = 0;
  size_t k = 0;
  size_t n = 0;
};
}  // namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi
