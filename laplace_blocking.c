#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define PI (3.1415926535897932384626)

int main(int argc, char **argv) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // For clarity, we print the number of processes being used
  if(rank == 0)
    printf("Using %d processes\n", size);

  // Initialize time measurement  
  double start_time = MPI_Wtime();

  int i, j, iter = 0;
  int n = 4096, m = 4096;

  // get runtime arguments (Useful to schedule multiple executions)
  if (argc > 1)
    n = atoi(argv[1]);
  if (argc > 2)
    m = atoi(argv[2]);
  
  // Calculate the number of rows each process will handle, this way the result is exactly the same
  // as if the matrix was divided evenly among the processes.
  // The first process will handle the extra rows if n is not divisible by size
  int base_rows = n / size;
  int extra_rows = n % size;

  int local_n;
  int first_row;

  if (rank < extra_rows) {
    local_n = base_rows + 1;
    first_row = rank * local_n;
  } else {
    local_n = base_rows;
    first_row = rank * base_rows + extra_rows;
  }

  // The data is dynamically allocated
  // We add rows above and below the local_n rows to handle boundaries
  float **A = (float **)malloc((local_n + 2) * sizeof(float *));
  float **Anew = (float **)malloc((local_n + 2) * sizeof(float *));
  for (i = 0; i < local_n + 2; i++) {
    A[i] = (float *)malloc(m * sizeof(float));
    Anew[i] = (float *)malloc(m * sizeof(float));
  }

  // All the interior points in the 2D matrix are zero
  for (i = 0; i < local_n + 2; i++)
    for (j = 0; j < m; j++)
      A[i][j] = 0;

  // set boundary conditions (left and right)
  for (i = 0; i < local_n + 2; i++) {
    int global_i = first_row - 1 + i;
    if (global_i >= 0 && global_i < n) { // Ensure top and bottom boundaries are 0
      A[i][0] = sinf(PI * global_i / (n - 1)); // Left boundary
      A[i][m - 1] = A[i][0] * expf(-PI); // Right boundary
    }
  }

  // If the maximum amount of change between two iterations is within
  // some tolerance, the outer loop will exit
  const float tol = 1.0e-3f; // Example tolerance (0.1%)
  float error = 1.0f;
  int iter_max = 1000; // Example
  // Main loop: iterate until error <= tol a maximum of iter_max iterations
  while (error > tol && iter < iter_max) {
    MPI_Request requests[4];
    int req_count = 0;
    // Synchronize the boundary rows with neighboring processes
    if (rank > 0) {
      MPI_Send(A[1], m, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);
      MPI_Recv(A[0], m, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
      MPI_Send(A[local_n], m, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);
      MPI_Recv(A[local_n + 1], m, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Calculate the new value for each element based on the current
    // values of its neighbors.
    for (i = 1; i <= local_n; i++)
      for (j = 1; j < m - 1; j++)
        Anew[i][j] = (A[i][j + 1] + A[i][j - 1] + A[i - 1][j] + A[i + 1][j]) / 4;

    // Compute local_error = maximum of the square root of the absolute differences
    // between the new value (Anew) and old one (A)
    float local_error = 0.0f;
    for (i = 1; i <= local_n; i++)
      for (j = 1; j < m - 1; j++)
        local_error = fmaxf(local_error, sqrtf(fabsf(Anew[i][j] - A[i][j])));

    // Update the value of A with the values calculated into Anew
    for (i = 1; i <= local_n; i++)
      for (j = 1; j < m - 1; j++)
        A[i][j] = Anew[i][j];
      
    MPI_Allreduce(&local_error, &error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD); // Reduce the error across all processes MPI Max

    // Every ten iterations the error must be printed
    iter++;
    if (rank == 0 && iter % 10 == 0)
      printf("%5d, %0.6f\n", iter, error);
  }
  double end_time = MPI_Wtime();
  if (rank == 0)
    printf("Total execution time: %f seconds\n\n", end_time - start_time);

  // Free allocated memory
  for (i = 0; i < local_n + 2; i++) {
    free(A[i]);
    free(Anew[i]);
  }
  free(A);
  free(Anew);

  MPI_Finalize();
  return 0;
}
