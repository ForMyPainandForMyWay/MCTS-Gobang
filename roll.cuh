#include"Node.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ int transform_GPU(int x, int y, int chess_length);
__global__ void roll_paralell(int* mat, int* step, int* step_nums, int* place, int* win_times, int* chess_len);
__device__ int judge_GPU(int* mat, int id, int put_place, int chess_len, int player);
void rollout_GPU(Node* node, int mat_size, const int* place, int place_eles, int roll_times);
