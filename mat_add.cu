#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template<typename T> struct DevBuf {
    T*  ptr = nullptr;
    size_t count = 0;

    explicit DevBuf(size_t n) : count(n) {
        cudaMalloc(&ptr, count * sizeof(T));
    }
    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr;
    }
    ~DevBuf() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    [[nodiscard]] size_t bytes() const { return count * sizeof(T); }
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "Error occured at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::abort(); \
        } \
    } while (0)

__global__ void matrix_add(
        const float* __restrict__ a, const float* __restrict__ b,
        float* __restrict__ c, int cols, int rows
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) {
        printf("Max: (%d, %d). Got: (%d, %d)\n", rows, cols, row, col);
        return;
    }

    int flat_idx = row * cols + col;
    c[flat_idx] = a[flat_idx] + b[flat_idx];
}

int main(int argc, char* argv[]) {
    constexpr int ROWS = 777;
    constexpr int COLS = 333;
    constexpr int N = ROWS * COLS;

    auto hA = new float[N];
    auto hB = new float[N];
    auto hC = new float[N];
    auto hRef = new float[N];
    for (int i = 0; i < N; i++) {
        hA[i] = static_cast<float>(1);
        hB[i] = static_cast<float>(2) * 2.f;
        hRef[i] = hA[i] + hB[i];
    }

    DevBuf<float> dA(N), dB(N), dC(N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, hA, dA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, hB, dB.bytes(), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
            (COLS + block.x - 1) / block.x,
            (ROWS + block.y - 1) / block.y
    );

    matrix_add<<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, COLS, ROWS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC.ptr, dC.bytes(), cudaMemcpyDeviceToHost));

    printf("Total elements: %d\n", N);
    printf("Threads generated: %d\n", (grid.x * grid.y) * 256) ;
    printf("Grid.x: %d | grid.y: %d | total blocks: %d\n", grid.x, grid.y, grid.x *  grid.y);
    printf("Mismatched thread-elem: %d\n", (grid.x * grid.y * 256 - N));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (hC[i] != hRef[i]) {
            printf("Mismatch at %d. Expected hRef[%d] = %f, got hC[%d] = %f\n",
                    i, i, hRef[i], i, hC[i]);
            if (++errors == 5) {
                break;
            }
        }
    }

    if (errors == 0)
        printf("Checked %d elements and no errors found!\n", N);

    delete[] hA; delete[] hB; delete[] hC; delete[] hRef;
    return (errors == 0) ? 0 : 1;
}
