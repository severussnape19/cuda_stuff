#include <cstdio>

__global__ void vector_scale(float *vec, float scalar, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        vec[idx] *= scalar;
    }
}

__global__ void element_wise_multiply(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        c[idx] = b[idx] * a[idx];
    }
}

void ex2() {
    int N = 100'003;
    int size = N * sizeof(float);

    auto h_A = (float*)malloc(size);
    auto h_B = (float*)malloc(size);
    auto h_C = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_A[i] = 2.0f;
        h_B[i] = 4.0f;
    }

    float *d_A;
    float *d_B;
    float *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    element_wise_multiply<<<grid_size, block_size>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 8.0f) {
            printf("Error at %d: h_C[%d] = %f\n", i, i, h_C[i]);
            if (++errors == 5) {
                break;
            }
        }
    }

    if (errors == 0) {
        printf("Checked %d elements. No errors found! h_C[0] = %f\n", N, h_C[0]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

void ex1() {
    int N = 500'000;
    int size = N * sizeof(float);

    auto h_a = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_a[i] = 2.0f;
    }

    float *d_a;
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    vector_scale<<<grid_size, block_size>>>(d_a, 3.0f, N);

    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_a[i] != 6.0) {
            printf("Error at %d: Got %f\n", i, h_a[i]);
            if (++errors == 5) break;
        }
    }

    if (errors == 0) {
        printf("checked %d elements. No errors occured! h_a[0] = %f\n", N, h_a[0]);
    }
    cudaFree(d_a);
    free(h_a);
}

int main() {
    ex2();
}
