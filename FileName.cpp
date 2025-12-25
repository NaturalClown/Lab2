#include <mpi.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <omp.h>

#define NOMINMAX
#include <windows.h>

const double ALPHA = 1e5;
const double EPS = 1e-8;
const int MAX_ITER = 100000;

const double X_MIN = -1.0, X_MAX = 1.0;
const double Y_MIN = -1.0, Y_MAX = 1.0;
const double Z_MIN = -1.0, Z_MAX = 1.0;

double exact_solution(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double right_part(double x, double y, double z) {
    return 6.0 - ALPHA * exact_solution(x, y, z);
}

double my_max(double a, double b) {
    return (a > b) ? a : b;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpi_prof_enabled = 0;
    MPI_Pcontrol(1); // Попытка профилирования
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int omp_threads = 1;
    if (argc >= 5) {
        omp_threads = atoi(argv[4]);
    }
    omp_set_num_threads(omp_threads);

    // Размеры сетки
    int Nx = 64, Ny = 64, Nz = 64; 
    if (argc >= 4) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
    }

    if (mpi_rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "HYBRID MPI+OpenMP Poisson Solver" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "MPI processes: " << mpi_size << std::endl;
        std::cout << "OpenMP threads per process: " << omp_threads << std::endl;
        std::cout << "Total threads: " << mpi_size * omp_threads << std::endl;
        std::cout << "Grid: " << Nx << " x " << Ny << " x " << Nz << std::endl;
        std::cout << "========================================" << std::endl;
    }

    double hx = (X_MAX - X_MIN) / (Nx - 1);
    double hy = (Y_MAX - Y_MIN) / (Ny - 1);
    double hz = (Z_MAX - Z_MIN) / (Nz - 1);

    double hx2 = hx * hx;
    double hy2 = hy * hy;
    double hz2 = hz * hz;
    double denominator = 2.0 / hx2 + 2.0 / hy2 + 2.0 / hz2 + ALPHA;

    int chunk_size = Nz / mpi_size;
    int remainder = Nz % mpi_size;

    int z_start, z_end, local_nz;

    if (mpi_rank < remainder) {
        local_nz = chunk_size + 1;
        z_start = mpi_rank * local_nz;
    }
    else {
        local_nz = chunk_size;
        z_start = remainder * (chunk_size + 1) + (mpi_rank - remainder) * chunk_size;
    }
    z_end = z_start + local_nz - 1;

    int total_layers = local_nz + 2;

    // Выделяем память
    std::vector<double> phi(Nx * Ny * total_layers, 0.0);
    std::vector<double> phi_new(Nx * Ny * total_layers, 0.0);
    std::vector<double> rho(Nx * Ny * total_layers, 0.0);

    auto idx = [&](int i, int j, int layer) {
        return (i * Ny + j) * total_layers + layer;
        };

    double init_start = MPI_Wtime();

#pragma omp parallel for
    for (int i = 0; i < Nx; i++) {
        double x = X_MIN + i * hx;
        for (int j = 0; j < Ny; j++) {
            double y = Y_MIN + j * hy;
            for (int layer = 0; layer < total_layers; layer++) {
                int global_z = z_start + layer - 1;

                if (global_z < 0 || global_z >= Nz) continue;

                double z = Z_MIN + global_z * hz;

                rho[idx(i, j, layer)] = right_part(x, y, z);
                phi[idx(i, j, layer)] = 0.0;

                if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 ||
                    global_z == 0 || global_z == Nz - 1) {
                    phi[idx(i, j, layer)] = exact_solution(x, y, z);
                }
            }
        }
    }

    double init_time = MPI_Wtime() - init_start;

    double compute_start = MPI_Wtime();
    int iter = 0;
    double global_diff = 1.0;

    std::vector<double> send_down(Nx * Ny);
    std::vector<double> send_up(Nx * Ny);
    std::vector<double> recv_down(Nx * Ny);
    std::vector<double> recv_up(Nx * Ny);

    MPI_Request requests[4];

    while (global_diff > EPS && iter < MAX_ITER) {
        double local_diff = 0.0;

        int req_count = 0;

        if (mpi_rank > 0) {
#pragma omp parallel for
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    send_down[i * Ny + j] = phi[idx(i, j, 1)];
                }
            }

            MPI_Isend(send_down.data(), Nx * Ny, MPI_DOUBLE, mpi_rank - 1, 0,
                MPI_COMM_WORLD, &requests[req_count++]);

            MPI_Irecv(recv_down.data(), Nx * Ny, MPI_DOUBLE, mpi_rank - 1, 1,
                MPI_COMM_WORLD, &requests[req_count++]);
        }

        if (mpi_rank < mpi_size - 1) {
#pragma omp parallel for
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    send_up[i * Ny + j] = phi[idx(i, j, local_nz)];
                }
            }

            MPI_Isend(send_up.data(), Nx * Ny, MPI_DOUBLE, mpi_rank + 1, 1,
                MPI_COMM_WORLD, &requests[req_count++]);

            MPI_Irecv(recv_up.data(), Nx * Ny, MPI_DOUBLE, mpi_rank + 1, 0,
                MPI_COMM_WORLD, &requests[req_count++]);
        }

#pragma omp parallel
        {
            double thread_diff = 0.0;

#pragma omp for nowait
            for (int i = 1; i < Nx - 1; i++) {
                for (int j = 1; j < Ny - 1; j++) {
                    for (int layer = 2; layer <= local_nz - 1; layer++) {
                        int index = idx(i, j, layer);

                        double sum = (phi[idx(i + 1, j, layer)] + phi[idx(i - 1, j, layer)]) / hx2
                            + (phi[idx(i, j + 1, layer)] + phi[idx(i, j - 1, layer)]) / hy2
                            + (phi[idx(i, j, layer + 1)] + phi[idx(i, j, layer - 1)]) / hz2;

                        phi_new[index] = (sum - rho[index]) / denominator;

                        double diff = fabs(phi_new[index] - phi[index]);
                        if (diff > thread_diff) thread_diff = diff;
                    }
                }
            }

#pragma omp critical
            {
                if (thread_diff > local_diff) local_diff = thread_diff;
            }
#pragma omp barrier

#pragma omp for nowait
            for (int i = 1; i < Nx - 1; i++) {
                for (int j = 1; j < Ny - 1; j++) {
                    int layer = 1;
                    int index = idx(i, j, layer);

                    double sum = (phi[idx(i + 1, j, layer)] + phi[idx(i - 1, j, layer)]) / hx2
                        + (phi[idx(i, j + 1, layer)] + phi[idx(i, j - 1, layer)]) / hy2
                        + (phi[idx(i, j, layer + 1)] + phi[idx(i, j, layer - 1)]) / hz2;

                    phi_new[index] = (sum - rho[index]) / denominator;

                    double diff = fabs(phi_new[index] - phi[index]);
                    if (diff > thread_diff) thread_diff = diff;
                }
            }

#pragma omp for
            for (int i = 1; i < Nx - 1; i++) {
                for (int j = 1; j < Ny - 1; j++) {
                    int layer = local_nz;
                    int index = idx(i, j, layer);

                    double sum = (phi[idx(i + 1, j, layer)] + phi[idx(i - 1, j, layer)]) / hx2
                        + (phi[idx(i, j + 1, layer)] + phi[idx(i, j - 1, layer)]) / hy2
                        + (phi[idx(i, j, layer + 1)] + phi[idx(i, j, layer - 1)]) / hz2;

                    phi_new[index] = (sum - rho[index]) / denominator;

                    double diff = fabs(phi_new[index] - phi[index]);
                    if (diff > thread_diff) thread_diff = diff;
                }
            }

#pragma omp critical
            {
                if (thread_diff > local_diff) local_diff = thread_diff;
            }
        }

        if (req_count > 0) {
            MPI_Status statuses[4];
            MPI_Waitall(req_count, requests, statuses);
        }

        // Обновляем ghost layers
        if (mpi_rank > 0) {
#pragma omp parallel for
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    phi[idx(i, j, 0)] = recv_down[i * Ny + j];
                }
            }
        }

        if (mpi_rank < mpi_size - 1) {
#pragma omp parallel for
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    phi[idx(i, j, local_nz + 1)] = recv_up[i * Ny + j];
                }
            }
        }

#pragma omp parallel for
        for (int i = 1; i < Nx - 1; i++) {
            for (int j = 1; j < Ny - 1; j++) {
                for (int layer = 1; layer <= local_nz; layer++) {
                    int index = idx(i, j, layer);
                    phi[index] = phi_new[index];
                }
            }
        }

        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        iter++;

        if (mpi_rank == 0 && iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", diff = " << global_diff << std::endl;
        }
    }

    double compute_time = MPI_Wtime() - compute_start;
    double total_time = compute_time + init_time;

    double local_max_error = 0.0;

#pragma omp parallel
    {
        double thread_max_error = 0.0;

#pragma omp for
        for (int i = 0; i < Nx; i++) {
            double x = X_MIN + i * hx;
            for (int j = 0; j < Ny; j++) {
                double y = Y_MIN + j * hy;
                for (int layer = 1; layer <= local_nz; layer++) {
                    int global_z = z_start + layer - 1;
                    double z = Z_MIN + global_z * hz;

                    double exact_val = exact_solution(x, y, z);
                    double computed = phi[idx(i, j, layer)];
                    double error = fabs(computed - exact_val);

                    if (error > thread_max_error) {
                        thread_max_error = error;
                    }
                }
            }
        }

#pragma omp critical
        {
            if (thread_max_error > local_max_error) {
                local_max_error = thread_max_error;
            }
        }
    }

    double global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double max_total_time, max_compute_time;
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "HYBRID MPI+OpenMP RESULTS:" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Configuration: " << mpi_size << " MPI x " << omp_threads << " OpenMP" << std::endl;
        std::cout << "Total threads: " << mpi_size * omp_threads << std::endl;
        std::cout << "Grid: " << Nx << " x " << Ny << " x " << Nz << std::endl;
        std::cout << "Iterations: " << iter << std::endl;
        std::cout << "Total time: " << max_total_time << " seconds" << std::endl;
        std::cout << "  Init time: " << init_time << " seconds" << std::endl;
        std::cout << "  Compute time: " << max_compute_time << " seconds" << std::endl;
        std::cout << "Time per iteration: " << max_compute_time / iter << " seconds" << std::endl;
        std::cout << "Final diff: " << global_diff << std::endl;
        std::cout << "Max error: " << global_max_error << std::endl;
        std::cout << "Converged: " << (global_diff <= EPS ? "YES" : "NO") << std::endl;
        std::cout << "========================================\n" << std::endl;

        std::ofstream out("hybrid_results.txt", std::ios::app);
        if (out) {
            out << mpi_size << " " << omp_threads << " "
                << max_total_time << " " << max_compute_time << " "
                << iter << " " << global_max_error << " "
                << global_diff << std::endl;
        }
    }
    if (mpi_rank == 0) {
        std::cout << "\n=== MPI Stat ===" << std::endl;
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        std::cout << "Processor: " << processor_name << std::endl;
    }

    if (mpi_prof_enabled) {
        MPI_Pcontrol(0);
    }
    MPI_Finalize();
    return 0;

}
