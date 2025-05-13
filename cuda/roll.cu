#include <mutex>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "roll.cuh"
using namespace std;
#define METHOD 5
#define NO_MAN (-1)
recursive_mutex mut2;


#define CHECK_KERNEL_CALL() do { \
cudaError_t err = cudaPeekAtLastError(); \
if (err != cudaSuccess) { \
fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
err = cudaStreamSynchronize(cudaStreamPerThread); \
if (err != cudaSuccess) { \
fprintf(stderr, "Kernel crashed: %s\n", cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while(0)


__device__ void cudaSwap(int& a, int& b) {
	a ^= b ^= a ^= b;
}

__device__ int transform_GPU(const int x, const int y, const int chess_length) {
	return x * chess_length + y;
}

__device__ int judge_GPU(const int* mat, const int input_place, const int chess_len, const int player) {
	const int max_bound = chess_len - 1;
	const int y = input_place % chess_len;
	const int x = input_place / chess_len;  // 更安全的除法写法

	// 四向增量：水平、垂直、正对角线、反对角线
	constexpr int dirs[4][2] = {{1,0}, {0,1}, {1,1}, {-1,1}};

	for (const auto dir : dirs) {
		const int dx = dir[0];
		const int dy = dir[1];
		int count = 1;  // 包含当前落子位置

		// 正向延伸检查
		for (int step = 1; ; ++step) {
			const int nx = x + dx * step;
			const int ny = y + dy * step;

			// 边界检查
			if (nx < 0 || nx > max_bound || ny < 0 || ny > max_bound) break;

			// 棋子匹配检查
			if (mat[transform_GPU(nx, ny, chess_len)] != player) break;

			++count;
		}

		// 反向延伸检查
		for (int step = 1; ; ++step) {
			const int nx = x - dx * step;
			const int ny = y - dy * step;

			// 边界检查
			if (nx < 0 || nx > max_bound || ny < 0 || ny > max_bound) break;

			// 棋子匹配检查
			if (mat[transform_GPU(nx, ny, chess_len)] != player) break;

			++count;
		}

		// 连珠条件
		if (count >= METHOD) return player;
	}

	return NO_MAN;
}

__global__ void roll_paralell(const int* __restrict__ mat, const int* __restrict__ step_nums,
							const int* __restrict__ place, int* __restrict__ win_times,
							const int* __restrict__ chess_len, const int* __restrict__ player_now,
							int* __restrict__ global_mat_buffer, int* __restrict__ global_step_buffer)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const int S = *chess_len * *chess_len;
    const unsigned int id = tid + bid * blockDim.x;

    // 通过全局内存缓冲区和线程索引获取本线程的存储空间
    int* mat_GPU = global_mat_buffer + id * S;
    int* step = global_step_buffer + id * (*step_nums);

    // 初始化棋盘副本
    for (int i = 0; i < S; i++)
        mat_GPU[i] = mat[i];

    // 初始化步骤序列
    for (int i = 0; i < *step_nums; i++)
        step[i] = place[i];

    // 洗牌
    curandState state;
    curand_init(clock64() + id, id, 0, &state);
    for (int i = *step_nums - 1; i >= 1; --i)
        cudaSwap(step[i], step[static_cast<int>(curand_uniform(&state) * *chess_len) % i]);

    // 模拟逻辑
    int win = NO_MAN;
    int index = 0;
    int player_now_local = *player_now;
    while (win == NO_MAN && index < *step_nums) {
        const int put_place = step[index++];
        mat_GPU[put_place] = player_now_local;
        win = judge_GPU(mat_GPU, put_place, *chess_len, player_now_local);
        player_now_local = (player_now_local + 1) % 2;
    }
    win_times[id] = win;
};


void rollout_GPU(Node* node,int mat_size,  const int* place, int place_eles, int roll_times) {
	// 预先判断是否可以rollout
	if (node->state->leaf == 1) {
		mut2.lock();
		// 对模拟结点进行反向传播
		node->update(1.0, 1.0);
		mut2.unlock();
		return;
	}

    constexpr int iDevice = 0;
    cudaSetDevice(iDevice);
	// 计算棋盘参数
	const int S = mat_size * mat_size;
	const int chess_len = node->state->chess->length;

    // Host存储结果
    const auto host_win_times = new int[roll_times];
    memset(host_win_times, 0, roll_times * sizeof(int));

    // Device内存
    int *device_place, *device_step_nums, *device_length, *player_now, *device_mat;
    int *devices_win_times;
    int *global_mat_buffer, *global_step_buffer;  // 全局缓冲区

	// 申请显存
    cudaMalloc((void**)&player_now, sizeof(int));
    cudaMalloc((void**)&device_mat, S * sizeof(int));
    cudaMalloc((void**)&device_place, place_eles * sizeof(int));
    cudaMalloc((void**)&devices_win_times, roll_times * sizeof(int));
    cudaMalloc((void**)&device_step_nums, sizeof(int));
    cudaMalloc((void**)&device_length, sizeof(int));

    // 分配全局缓冲区
    cudaMalloc(&global_mat_buffer, roll_times * S * sizeof(int));
    cudaMalloc(&global_step_buffer, roll_times * place_eles * sizeof(int));

    // 数据拷贝
    cudaMemcpyAsync(player_now, &node->state->chess->who_place_next, sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(device_mat, node->state->chess->mat, S * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(device_place, place, place_eles * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(device_step_nums, &place_eles, sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread);
    cudaMemcpyAsync(device_length, &chess_len, sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread);

    // 运行核函数,随机落子
    int block = 32;
    int grid = (roll_times + block - 1) / block;
    roll_paralell<<<grid, block>>>(device_mat, device_step_nums, device_place,
                                  devices_win_times, device_length, player_now,
                                  global_mat_buffer, global_step_buffer);
    CHECK_KERNEL_CALL();

    // 结果回传
    cudaMemcpyAsync(host_win_times, devices_win_times, roll_times * sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread);

    // 统计结果
    float win_count = 0;
    for (int i = 0; i < roll_times; i++)
        if (host_win_times[i] == 1) win_count++;

    // 更新节点
	{
		lock_guard lock(mut2);
		node->update(1, win_count / roll_times);
	}

    // 资源释放
    cudaFree(global_mat_buffer);
    cudaFree(global_step_buffer);
    cudaFree(player_now);
    cudaFree(device_mat);
    cudaFree(devices_win_times);
    cudaFree(device_place);
    cudaFree(device_step_nums);
    cudaFree(device_length);
    delete[] host_win_times;
}