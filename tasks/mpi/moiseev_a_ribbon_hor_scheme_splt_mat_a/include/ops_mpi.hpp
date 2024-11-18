#include <boost/mpi/collectives.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>

#include "core/task/include/task.hpp"

namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi {

template <typename DataType>
class MatrixMultiplicationParallel : public ppc::core::Task {
 public:
  explicit MatrixMultiplicationParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(taskData_), taskData(taskData_) {}

  bool pre_processing() override {
    internal_order_test();

    if ((taskData->inputs.size() < 2) || taskData->inputs_count.size() != 3) {
      return false;
    }

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
    return (taskData->inputs.size() == 2 && taskData->inputs_count.size() == 3 && m * k == n * k);
  }

bool run() override {
    internal_order_test();

    // MPI-коммуникатор и параметры ранга
    boost::mpi::communicator world;
    int rank = world.rank();
    int size = world.size();

    // Делим строки матрицы A между процессами
    size_t rows_per_proc = m / size;
    size_t remainder = m % size;
    size_t start_row = rank * rows_per_proc + std::min(static_cast<size_t>(rank), remainder);
    size_t end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);
    size_t local_row_count = end_row - start_row;

    //std::cout << "Number of processes: " << world.size() << ", Current process rank: " << rank << "\n";

    // Распределяем строки матрицы A между процессами
    std::vector<DataType> local_A(local_row_count * k);
    if (rank == 0) {
      std::vector<int> sendcounts(size), displs(size);
      for (int i = 0; i < size; ++i) {
        size_t proc_start_row = i * rows_per_proc + std::min(static_cast<size_t>(i), remainder);
        size_t proc_row_count = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = proc_row_count * k;
        displs[i] = proc_start_row * k;
      }
      boost::mpi::scatterv(world, A.data(), sendcounts, displs, local_A.data(), local_A.size(), 0);

      // Отладочный вывод матрицы A на главном процессе
      //std::cout << "Process " << rank << ": Matrix A (global):\n";
      //for (size_t i = 0; i < m; ++i) {
      //  for (size_t j = 0; j < k; ++j) {
      //    std::cout << A[i * k + j] << " ";
      //  }
      //  std::cout << "\n";
      //}
    } else {
      boost::mpi::scatterv(world, A.data(), {}, {}, local_A.data(), local_A.size(), 0);
    }

    // Рассылаем матрицу B всем процессам
    boost::mpi::broadcast(world, B, 0);
    //if (rank == 0) {
    //  // Отладочный вывод матрицы B
    //  std::cout << "Process " << rank << ": Matrix B (global):\n";
    //  for (size_t i = 0; i < k; ++i) {
    //    for (size_t j = 0; j < n; ++j) {
    //      std::cout << B[i * n + j] << " ";
    //    }
    //    std::cout << "\n";
    //  }
    //}

    // Локальная часть матрицы C
    std::vector<DataType> local_C(local_row_count * n, 0);

    // Вычисление локальной части матрицы C
    for (size_t i = 0; i < local_row_count; ++i) {
      for (size_t j = 0; j < n; ++j) {
        for (size_t p = 0; p < k; ++p) {
          local_C[i * n + j] += local_A[i * k + p] * B[p * n + j];
        }
      }
    }

    // Отладочный вывод локальных данных матрицы A и результата C
    //std::cout << "Process " << rank << ": Local matrix A:\n";
    //for (size_t i = 0; i < local_row_count; ++i) {
    //  for (size_t j = 0; j < k; ++j) {
    //    std::cout << local_A[i * k + j] << " ";
    //  }
    //  std::cout << "\n";
    //}
    //std::cout << "Process " << rank << ": Local matrix C:\n";
    //for (size_t i = 0; i < local_row_count; ++i) {
    //  for (size_t j = 0; j < n; ++j) {
    //    std::cout << local_C[i * n + j] << " ";
    //  }
    //  std::cout << "\n";
    //}

    // Сбор локальных частей C на главном процессе
    if (rank == 0) {
      std::vector<int> recvcounts(size), displs(size);
      for (int i = 0; i < size; ++i) {
        size_t proc_start_row = i * rows_per_proc + std::min(static_cast<size_t>(i), remainder);
        size_t proc_row_count = rows_per_proc + (i < remainder ? 1 : 0);
        recvcounts[i] = proc_row_count * n;
        displs[i] = proc_start_row * n;
      }
      boost::mpi::gatherv(world, local_C.data(), local_C.size(), C.data(), recvcounts, displs, 0);

      // Отладочный вывод глобальной матрицы C
      //std::cout << "Process " << rank << ": Matrix C (global):\n";
      //for (size_t i = 0; i < m; ++i) {
      //  for (size_t j = 0; j < n; ++j) {
      //    std::cout << C[i * n + j] << " ";
      //  }
      //  std::cout << "\n";
      //}
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
  std::vector<DataType> A, B, C;
  size_t m = 0, k = 0, n = 0;
};
}  // namespace moiseev_a_ribbon_hor_scheme_splt_mat_a_mpi
