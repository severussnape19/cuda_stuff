#include <array>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

template<typename T> struct DevBuf {
    T*     ptr = nullptr;
    size_t count = 0;

    explicit DevBuf (size_t n) : count(n) {
        cudaMalloc(&ptr, count * sizeof(T));
    }
    ~DevBuf() {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), count(o.count) {
        o.ptr = nullptr;
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
        } \
    } while (0);

__global__ void scalar_multiply(
        float* __restrict__ mat, float alpha,
        int rows, int cols
        )
{
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row >= rows || col >= cols) return;

    size_t idx = row * cols + col;
    mat[idx] *= alpha;
}

__global__ void matrix_transpose(
        const float* __restrict__ a, float* __restrict__ b,
        int src_rows, int src_cols
        )
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= src_rows || col >= src_cols) return;

    int val = a[row * src_cols + col]; // Row major
    b[col * src_rows + row] = val; // Column major
}

int vec_scale() {
    constexpr size_t ROWS = 500;
    constexpr size_t COLS = 700;
    constexpr size_t N = ROWS * COLS;

    std::array<float, 4> alpha_vals = { 0.0f, 1.0f, -1.0f, 3.14159f };

    auto hRef1 = new float[N];
    auto hRef2 = new float[N];
    auto hRef3 = new float[N];
    auto hRef4 = new float[N];
    for (int i = 0; i < N; i++) {
        hRef1[i] = static_cast<float>(i) * alpha_vals[0];
        hRef2[i] = static_cast<float>(i) * alpha_vals[1];
        hRef3[i] = static_cast<float>(i) * alpha_vals[2];
        hRef4[i] = static_cast<float>(i) * alpha_vals[3];
    }

    dim3 block(16, 16);
    dim3 grid(
            (COLS + block.x - 1) / block.x,
            (ROWS + block.y - 1) / block.y
    );

    printf("Info:\nRows: %lu | Cols: %lu | N: %lu\nBlockDim: 16 * 16 | Blocks: %d\nGrid.x: %d | Grid.y: %d\nRequired Threads: %lu | Generated threads: %u | Excess Threads: %lu\n\n",
            ROWS, COLS, N, (grid.x * grid.y), grid.x, grid.y, N, (grid.x * grid.y) * 256, (grid.x * grid.y) * 256 - N);

    std::array<float*, 4> references = { hRef1, hRef2, hRef3, hRef4 };
    size_t ref_idx = 0;

    for (auto alpha : alpha_vals) {
        printf("Alpha: %f\n", alpha);

        auto hA = new float[N];
        for (int i = 0; i < N; i++) {
            hA[i] = static_cast<float>(i);
        }

        DevBuf<float> dA(N);
        CUDA_CHECK(cudaMemcpy(dA.ptr, hA, dA.bytes(), cudaMemcpyHostToDevice));
        scalar_multiply<<<grid, block>>>(dA.ptr, alpha, ROWS, COLS);
        CUDA_CHECK(cudaMemcpy(hA, dA.ptr, dA.bytes(), cudaMemcpyHostToDevice));

        int errors = 0;
        for (int i = 0; i < N; i++) {
            if (hA[i] != references[ref_idx][i]) {
                printf("Error at %d. Expected %f Got %f\n", i, references[ref_idx][i], hA[i]);
                if (++errors > 5) {
                    break;
                }
            }
        }

        if (errors == 0) {
            printf("Iterated over %lu elements. No errors found!\n", N);
        }
        ref_idx++;
        delete[] hA;
    }

    delete[] hRef1; delete[] hRef2; delete[] hRef3; delete[] hRef4;
    return 0;
}

int main() {
    constexpr size_t ROWS = 1024;
    constexpr size_t COLS = 512;
    constexpr size_t N = ROWS * COLS;

    auto hSrc = new float[N];
    auto hDest = new float[N];
    auto hRef = new float[N];

    float val = 0.f;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            size_t idx = i * COLS + j;
            hSrc[idx] = static_cast<float>(val++);
        }
    }

    for (int row = 0; row < ROWS; row++) {
        for (int col = 0; col < COLS; col++) {
            size_t ref_idx = col * ROWS + row;
            size_t src_idx = row * COLS + col;
            hRef[ref_idx] = hSrc[src_idx];
        }
    }

    DevBuf<float> dSrc(N), dDest(N);
    CUDA_CHECK(cudaMemcpy(dSrc.ptr, hSrc, dSrc.bytes(), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(
            (COLS + block.x - 1) / block.x,
            (ROWS + block.y - 1) / block.y
    );

    matrix_transpose<<<grid, block>>>(dSrc.ptr, dDest.ptr, ROWS, COLS);
    CUDA_CHECK(cudaMemcpy(hDest, dDest.ptr, dDest.bytes(), cudaMemcpyHostToDevice));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (hDest[i] != hRef[i]) {
            printf("Error at %ld! Expected %f. Got %f\n", N, hRef[i], hDest[i]);
            if (++errors > 5) {
                break;
            }
        }
    }

    if (errors == 0) {
        printf("Checked %ld elements! no errors found!!!\n", N);
    }

    delete[] hRef; delete[] hDest; delete[] hSrc;
    return 0;
}
