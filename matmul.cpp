#include <immintrin.h>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstring>
#include <chrono>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

// Define tile config data structure 
struct __tilecfg {
    // palette_id
    // 0: initial state 
    // 1: matmal 
    // >=2: future ops 
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[8];          // width of tile 0-7 in bytes
    uint16_t reserved_colsb[8]; // future tiles  
    uint8_t rows[8];            // rows of tile 0-7
    uint8_t reserved_rows[8];   // future tiles
};

// Initialize tile config
void init_tile_config(__tilecfg *tileinfo) {
    tileinfo->palette_id = 1;   // palette 1: do matmal
    tileinfo->start_row = 0;

    // Special tile 0
    tileinfo->colsb[0] = MAX_ROWS;
    tileinfo->rows[0] = MAX_ROWS;

    // dst: tile_1, src1: tile_2, src2: tile_3
    for (int i = 1; i < 4; ++i) {
        tileinfo->colsb[i] = MAX_COLS;  // 64
        tileinfo->rows[i] = MAX_ROWS;   // 16
    }

    _tile_loadconfig(tileinfo);
}

// Initialize int8_t buffer
void init_buffer(int8_t *buf, int8_t value) {
    for (int i = 0; i < MAX_ROWS; ++i) {
        for (int j = 0; j < MAX_COLS; ++j) {
            buf[i * MAX_COLS + j] = value;
        }
    }
}

// Initialize int32_t buffer
void init_buffer32(int32_t *buf, int32_t value) {
    for (int i = 0; i < MAX_ROWS; ++i) {
        for (int j = 0; j < MAX_COLS / 4; ++j) {
            buf[i * (MAX_COLS / 4) + j] = value;
        }
    }
}

template <typename T>
void print_buffer(T* buf, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << static_cast<int>(buf[i * cols + j]) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE
bool set_tiledata_use() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        std::cout << "\n Fail to do XFEATURE_XTILEDATA \n\n";
        return false;
    } else {
        std::cout << "\n TILE DATA USE SET - OK \n\n";
        return true;
    }
}

// Perform matrix multiplication using tiles
void matmul(__tilecfg &tile_data, int8_t *src1, int8_t *src2, int32_t *dst) {
    if (!set_tiledata_use()) {
        exit(-1);
    }

    init_tile_config(&tile_data);

    _tile_loadd(2, src1, STRIDE);
    _tile_loadd(3, src2, STRIDE);
    _tile_loadd(1, dst, STRIDE);

    _tile_dpbssd(1, 2, 3);

    _tile_stored(1, dst, STRIDE);
}

int min(int a, int b) {
    return (a > b) ? b : a;
}

// Helper function to perform arbitrary matrix multiplication
void matmul_blocked(__tilecfg &tile_data, int8_t *A, int8_t *B, int32_t *C, int M, int N, int K) {
    const int BLOCK_M = 16;
    const int BLOCK_N = 16;
    const int BLOCK_K = 64;

    for (int i = 0; i < M; i += BLOCK_M) {
        for (int j = 0; j < N; j += BLOCK_N) {
            for (int k = 0; k < K; k += BLOCK_K) {
                int m_block = min(BLOCK_M, M - i);
                int n_block = min(BLOCK_N, N - j);
                int k_block = min(BLOCK_K, K - k);

                int8_t A_block[BLOCK_M * BLOCK_K] = {};
                int8_t B_block[BLOCK_K * BLOCK_N] = {};
                int32_t C_block[BLOCK_M * BLOCK_N] = {};

                for (int m = 0; m < m_block; ++m) {
                    for (int kk = 0; kk < k_block; ++kk) {
                        A_block[m * BLOCK_K + kk] = A[(i + m) * K + (k + kk)];
                    }
                }

                for (int kk = 0; kk < k_block; ++kk) {
                    for (int n = 0; n < n_block; ++n) {
                        B_block[kk * BLOCK_N + n] = B[(k + kk) * N + (j + n)];
                    }
                }

                matmul(tile_data, A_block, B_block, C_block);

                for (int m = 0; m < m_block; ++m) {
                    for (int n = 0; n < n_block; ++n) {
                        C[(i + m) * N + (j + n)] += C_block[m * BLOCK_N + n];
                    }
                }
            }
        }
    }
}

int main() {
    __tilecfg tile_data = {0};
    int8_t src1[MAX];
    int8_t src2[MAX];
    int32_t dst[MAX / 4];
    int rows = MAX_ROWS;
    int colsb = MAX_COLS;
    
    int8_t v = 2; 
    init_buffer(src1, v);
    init_buffer(src2, v);
    init_buffer32(dst, 0);
    init_tile_config(&tile_data);

    matmul(tile_data, src1, src2, dst);

    int M = 128;
    int N = 64;
    int K = 64;

    int8_t A[M * K];
    int8_t B[K * N];
    int32_t C[M * N];
    memset(A, 2, sizeof(A));
    memset(B, 2, sizeof(B));
    memset(C, 0, sizeof(C));

    // warm up
    for (int k=0; k<10; ++k) {
        matmul_blocked(tile_data, A, B, C, M, N, K);
    }
    memset(A, 2, sizeof(A));
    memset(B, 2, sizeof(B));
    memset(C, 0, sizeof(C));

    int exectime = 100;
    auto start = std::chrono::system_clock::now();
    for (int k=0; k<exectime; ++k) {
        matmul_blocked(tile_data, A, B, C, M, N, K);
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "avg_infer_time = " << double(duration.count()) / 1e3 / exectime << " us" << std::endl;
    print_buffer(C, M, N);
    // 79.4725 us
    // 0.976562 us

    _tile_release();
    return 0;
}
