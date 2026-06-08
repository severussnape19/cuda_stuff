#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

template<typename T> struct DevBuf 
{
    T* ptr{nullptr};
    size_t count{0uz};

    explicit DevBuf(size_t n) : count(n) {
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

    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(const DevBuf&) = delete;

    [[nodiscard]] auto bytes() const -> size_t { return count * sizeof(T); }
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "Error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
            std::abort(); \
        } \
    } while (0);

/* The first thread executes 2 times. Once at 0 and then jumps to `stride` elements from idx.
stride = BlockDim.x * gridDim.x (1D consecutive elements)Thats equal to 256 * (32 * 24) = 196608. So Most elements would
just be accessed once. 777 * 333 = 258,743. 258741-196609 = 62133. First 62133 thrreads would execute twice and the rest just once.

Or incase of 256 blocks with 256 threads, we would have 65536 threads in total. 
Thread 0 starts at 0. 
Then 0 + 65536 = 65536 | 258741 > 65536
Then 65546 + 65536 = 131072. | 258741 > 131072
Then 131072 + 65536 = 196608 | greater than N
196608 + 65536 = 262144 | Exceeds N. Breaks.

The last thread would just run once. Cuz i incremented by even 1 is out of bounds.
*/
__global__ auto mat_add_grid_stride(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) -> void 
{
    size_t idx{ blockIdx.x * blockDim.x + threadIdx.x };
    size_t stride{ gridDim.x * blockDim.x }; // Totoal threads

    for (auto i{idx}; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__global__ auto saxpy_grid_stride(
    const float* __restrict__ A,
    float* __restrict__ B,
    float alpha,
    int N
) -> void 
{
    size_t idx{ blockDim.x * blockIdx.x + threadIdx.x };
    size_t stride{ gridDim.x * blockDim.x };

    for (auto i{idx}; i < N; i += stride) {
        B[i] = alpha * A[i] + B[i];
    }
}

__global__ auto relu_grid_stride(float* __restrict__ A, int N ) -> void 
{
    size_t idx{ blockIdx.x * blockDim.x + threadIdx.x };
    size_t stride{ gridDim.x * blockDim.x };

    for (auto i{idx}; i < N; i += stride) {
        A[i] = std::fmax(0.0f, A[i]);
    }
}

__global__ auto strided_copy_(
    const float* __restrict__ src,
    float* __restrict__ dst,
    size_t src_len,
    size_t K
) -> void
{
    auto dest_len{ src_len / K };
    size_t idx{ blockIdx.x * blockDim.x + threadIdx.x };
    size_t stride{ blockDim.x * gridDim.x };

    for (auto i{idx}; i < dest_len; i += stride) {
        dst[i] = src[i * K];
    }
}

auto strided_copy() -> int
{
    constexpr auto N_1{1'000'000uz};
    constexpr auto K{7uz};
    constexpr auto N_2{ N_1 / K };

    std::vector<float> hA(N_1);
    std::vector<float> hB(N_2);
    std::vector<float> hRef(N_2);
    auto j{0uz};
    for (auto i{0uz}; i < N_1; ++i) {
        hA[i] = static_cast<float>(i);
        if (i % K == 0 && j < N_2) {
            hRef[j++] = static_cast<float>(i);
        } 
    }

    DevBuf<float> dA(N_1), dB(N_2);
    CUDA_CHECK(cudaMemcpy(dA.ptr, hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, hB.data(), dB.bytes(), cudaMemcpyHostToDevice));

    auto threads{256uz};
    auto blocks{ std::min((N_2 + threads - 1) / threads, static_cast<size_t>(24 * 32)) };
    strided_copy_<<<threads, blocks>>>(dA.ptr, dB.ptr, N_1, K);
    CUDA_CHECK(cudaMemcpy(hB.data(), dB.ptr, dB.bytes(), cudaMemcpyDeviceToHost));

    auto errors{0uz};
    for (auto i{0uz}; i < N_2; ++i) {
        if (hRef[i] != hB[i]) {
            fprintf(stderr, "Errort at %lu. Expected %f. Got %f\n", 
                i, hRef[i], hB[i]);
            if (++errors > 5) {
                break;
            }
        }
    }
    
    if (errors == 0) {
        fprintf(stderr, "Checked %d elements. No errors\n", N_2);
    }

    return (errors == 0) ? 0 : 1;
}

auto relu() -> int 
{
    constexpr auto N{100'000uz};

    std::vector<float> hA(N);
    std::vector<float> hRef(N);
    
    for (auto i{0uz}; i < N; ++i) {
        if (i % 2 == 0) {
            hA[i] = static_cast<float>(i);
        } else {
            hA[i] = static_cast<float>(-i);
        }
        hRef[i] = std::fmax(0.0f, hA[i]);
    }

    DevBuf<float> dA(N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, hA.data(), dA.bytes(), cudaMemcpyHostToDevice));

    const auto threads{256uz};
    const auto blocks{ std::min( (N + threads - 1 ) / threads, static_cast<size_t>(24 * 32)) };
    relu_grid_stride<<<blocks, threads>>>(dA.ptr, N);
    CUDA_CHECK(cudaMemcpy(hA.data(), dA.ptr, dA.bytes(), cudaMemcpyDeviceToHost));

    auto errors{0uz};
    for (auto i{0uz}; i < N; ++i) {
        if (hA[i] != hRef[i]) {
            fprintf(stderr, "Errort at %lu. Expected %f. Got %f\n", 
                i, hRef[i], hA[i]);
            if (++errors > 5) {
                break;
            }
        }
    }

    if (errors == 0) {
        fprintf(stderr, "Checked %d elements. No errors\n", N);
    }

    return (errors == 0) ? 0 : 1;
}

/* With N = 100,000,000 elemnts and a grid of  256 * 256 threads, the first thread would run for 
N / gridDim.x * blockDim.x times. So 1525 times.
The stride value is 65536 if we take 256 * 256 dim.
Thread 37 processes 37, 65573,131109 elements.*/
auto saxpy(void) -> int
{
    constexpr auto N{100'000'000uz};
    const auto alpha{3.14f};

    std::vector<float> hA(N);
    std::vector<float> hB(N);
    std::vector<float> hRef(N);
    for (auto i{0uz}; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(i) * 2.f;
        hRef[i] = alpha * hA[i] + hB[i];
    }

    DevBuf<float> dA(N), dB(N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, hB.data(), dB.bytes(), cudaMemcpyHostToDevice));

    const auto threads{256uz};
    const auto blocks{ min(((N + threads - 1) / threads), static_cast<size_t>(32 * 24)) };

    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    saxpy_grid_stride<<<256, threads>>>(dA.ptr, dB.ptr, alpha, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    auto ms{0.f};
    cudaEventElapsedTime(&ms, start, stop);
    
    auto seconds{ ms * 1e-3f };
    auto bytes{ static_cast<float>(N) * 12.f };
    printf("N = %lu. %.3f ms\n", N, ms);
    
    auto bandwidth{ (bytes / seconds) * 1e-9f };
    auto efficiency{ (bandwidth / 272.0f) * 100.f };
    printf("Bandwidth: %f | Efficiency: %f\n", bandwidth, efficiency);

    CUDA_CHECK(cudaMemcpy(hB.data(), dB.ptr, dB.bytes(), cudaMemcpyDeviceToHost));

    auto errors{0uz};
    for (auto i{0uz}; i < N; ++i) {
        float absolute_diff{ std::abs(hB[i] - hRef[i]) };
        float dynamic_epsilon{ std::max(1e-3f, 1e-5f * std::abs(hRef[i])) };
        
        if (absolute_diff > dynamic_epsilon) {
            fprintf(stderr, "Error at %lu. Expected %f. Got %f. Diff(%f)\n",
                 i, hRef[i], hB[i], absolute_diff);
            
            if (++errors > 5) break;
        }
    }

    if (errors == 0) {
        fprintf(stderr, "Checked %lu elements! No errors found!\n", N);
    }

    return (errors == 0) ? 0 : 1;
}

auto mat_add(void) -> int 
{
    constexpr auto ROWS{777uz};
    constexpr auto COLS{333uz};
    constexpr auto N{ ROWS * COLS };

    std::vector<float> hA(N);
    std::vector<float> hB(N);
    std::vector<float> hC(N);
    std::vector<float> hRef(N);

    float val = 1.f;
    for (auto i{0uz}; i < N; ++i) {
        hA[i] = val;
        hB[i] = val * 1.2f;
        hRef[i] = hA[i] + hB[i];
    }

    DevBuf<float> dA(N), dB(N), dC(N);
    CUDA_CHECK(cudaMemcpy(dA.ptr, hA.data(), dA.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB.ptr, hB.data(), dB.bytes(), cudaMemcpyHostToDevice));

    const auto threads{ 256uz }; // block size
    const auto blocks{ min((N + threads - 1) / threads, static_cast<size_t>(24 * 32)) }; // Grid size

    printf("Threads: %lu | blocks: %lu\n", threads * blocks, blocks);

    mat_add_grid_stride<<<blocks, threads>>>(dA.ptr, dB.ptr,dC.ptr, N);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC.ptr, dC.bytes(), cudaMemcpyDeviceToHost));

    int errors{0};
    for (auto i{0uz}; i < N; ++i) {
        if (hC[i] != hRef[i]) {
            fprintf(stderr, "Error at %lu. Expected %f. Got %d\n", i, hRef[i], hC[i]);
            if (++errors > 5) {
                break;
            }
        }
    }

    if (errors == 0) {
        printf("Checked %lu elemets and found no errors!\n", N);
    }

    return (errors == 0) ? 0 : 1;
}

auto main() -> int 
{
    saxpy();
}