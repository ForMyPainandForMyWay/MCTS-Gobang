#include <mutex>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "roll.cuh"


using namespace std;
#define METHOD 5
#define NO_MAN (-1)
recursive_mutex mut2;

__device__ void cudaSwap(int& a, int& b) {
	a ^= b;
	b ^= a;
	a ^= b;
}

__device__ int transform_GPU(const int x, const int y, const int chess_length) {
	return x * chess_length + y;
}

__device__ int judge_GPU(const int* mat, const int input_place, const int chess_len, const int player) {
	const int max_bound = chess_len - 1;
	const int y = input_place % chess_len;
	const int x = input_place / chess_len;  // 更安全的除法写法

	// 四个方向的增量：水平、垂直、正对角线、反对角线
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

		// 判断是否满足连珠条件
		if (count >= METHOD) {
			return player;
		}
	}

	return NO_MAN;
}




__global__ void roll_paralell(const int* mat,const int* step_nums,const int* place,int* win_times,const int *chess_len,const int *player_now)
{
	int win = -1, index = 0;  // 是否赢，落子次数，落子位置索引，子节点第一个模拟的是player_0
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x;
	const int S = *chess_len * *chess_len;  // 计算棋盘面积
	const unsigned int id = tid + bid * blockDim.x;  // 获取线程索引
	//int base_index = id * S;  // 基地址
	const auto mat_GPU = new int[S];  // 创建棋盘副本
	int player_now_local = *player_now;

	// 复制棋盘内容
	for (int i = 0; i < S; i++) {
		mat_GPU[i] = mat[i];
	}

	// 构造随机落子位置序列
	int* step = new int[*step_nums];
	curandState state;
	curand_init(clock64() + id, id, 0, &state); // 使用相同的种子和不同的序列数来初始化
	for (int i = 0; i < *step_nums; i++) step[i] = place[i];
	for (int i = *step_nums - 1; i >= 1; --i) cudaSwap(step[i], step[int(curand_uniform(&state)* *chess_len) % i]);
	
	// 随机落子
	while (win == -1 && index < *step_nums) {
		const int put_place = step[index];
		index++;  // 落子次数、下棋位置索引  自增
		mat_GPU[put_place] = player_now_local;  //落子
		win = judge_GPU(mat_GPU, put_place, *chess_len, player_now_local);  // 裁决
		player_now_local = (player_now_local + 1) % 2;  // 切换下棋方
	}
	win_times[id] = win;
	delete[] step;
	delete[] mat_GPU;
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

	int iDevice = 0;
	cudaSetDevice(iDevice);

	// 将记录结果用的数字改为数组
	auto *host_win_times = new float[roll_times];
	memset(host_win_times, 0, roll_times * sizeof(float));

	// 创建所有线程的落子顺序序列，每个线程对应一定的范围(已废弃，改为由GPU自己创建)
	/*int* host_step = new int[roll_times * place_eles];
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	for (int l = 0; l < roll_times; l++) {
		for (int i = 0; i < place_eles; i++) {
			for (int j = 0; j < roll_times; j++) host_step[j + l * place_eles] = place[j];
			//for (int k = roll_times; k >= 1; --k) swap(host_step[k + l * place_eles], host_step[rand() % k]);
		}
	}*/

	// 提前将原来的棋盘拷贝多份到一个数组中,用于每一个线程各自索引(已废弃，改为由GPU自行构建副本)
	const int S = mat_size* mat_size;
	const int mat_len = S;
	//int* host_mat = new int[S];
	/*for (int i = 0; i < roll_times; i++) {
		for (int j = 0; j < S; j++) {
			cout << "i: " << i << "  j: " << j << endl;
			host_mat[j + i * S] = node->state->chess->mat[j];
		}
	}*/

	// 定义device数据(棋盘和落子位置),分配内存空间
	int *device_place,*device_step_nums, *device_length, *player_now, *device_mat;
	int* devices_win_times;
	// 直接调用 cudaMalloc，移除错误检查包装
	cudaMalloc((void**)&player_now, sizeof(int));
	cudaMalloc((void**)&device_mat, sizeof(int) * mat_len);
	cudaMalloc((void**)&device_place, sizeof(int) * place_eles);
	cudaMalloc((void**)&devices_win_times, sizeof(int) * roll_times);
	cudaMalloc((void**)&device_step_nums, sizeof(int));
	cudaMalloc((void**)&device_length, sizeof(int));

	// 初始化内存空间
	cudaMemset(player_now, 0, sizeof(int));
	cudaMemset(device_mat, 0, sizeof(int) * mat_len);
	cudaMemset(device_place, 0, sizeof(int) * place_eles);
	cudaMemset(devices_win_times, 0, sizeof(int) * roll_times);
	cudaMemset(device_step_nums, 0, sizeof(int));
	cudaMemset(device_length, 0, sizeof(int));

	// 将host数据与device进行拷贝
	cudaMemcpy(player_now, &node->state->chess->who_place_next, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_mat, node->state->chess->mat, sizeof(int) * mat_len, cudaMemcpyHostToDevice);
	cudaMemcpy(device_place, place, sizeof(int) * place_eles, cudaMemcpyHostToDevice);
	cudaMemcpy(devices_win_times, host_win_times, sizeof(int) * roll_times, cudaMemcpyHostToDevice);
	cudaMemcpy(device_step_nums, &place_eles, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_length, &node->state->chess->length, sizeof(int), cudaMemcpyHostToDevice);

	// 运行核函数,随机落子
	int block = 32;
	int grid = (roll_times + block - 1) / block;
	roll_paralell <<<grid, block>>> (device_mat, device_step_nums, device_place, devices_win_times, device_length, player_now);
	cudaDeviceSynchronize();

	// 将结果拷贝到host
	cudaMemcpy(host_win_times, devices_win_times, sizeof(float) * roll_times, cudaMemcpyDeviceToHost);
	
	// 统计结果
	float win_times = 0;
	for (int i = 0; i < roll_times; i++) if (host_win_times[i] == 1) win_times++;


	// 更新结点
	mut2.lock();
	node->update(1, win_times /  static_cast<float>(roll_times));
	mut2.unlock();

	// 释放资源
	cudaFree(player_now);
	cudaFree(device_mat);
	cudaFree(devices_win_times);
	cudaFree(device_place);
	cudaFree(device_step_nums);
	cudaFree(device_length);
	delete[] host_win_times;
	//delete[] host_step; 
	//delete[] host_mat;
}