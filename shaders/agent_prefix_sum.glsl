#[compute]
#version 450

// 单工作组并行 exclusive scan：按 tile（1024 线程 × 4 格）循环，线程 t 负责 tile 内第 4t..4t+3 格。
// 全局读写经 shared 中转以保持合并访问（每线程直接顺序扫一整段时写不合并，实测慢约 4 倍）。
// 前提：maxComputeWorkGroupInvocations >= 1024（RTX 5070 = 1024）。
// shared memory ≈ 20.3 KB（s_v 16 KB + s_sum 4.1 KB + s_blk 132 B）：超过 Vulkan 保证的 16 KB 下限，RTX 5070 为 48 KB，桌面 GPU 普遍 >= 32 KB。
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint cell_count[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer B1 { uint cell_start[]; };
layout(set = 0, binding = 2, std430) restrict writeonly buffer B2 { uint cell_offset[]; };

layout(push_constant, std430) uniform Params {
	int num_cells;
	int _p0; int _p1; int _p2;
};

const uint WG = 1024u;
const uint TILE = WG * 4u;
shared uvec4 s_v[WG];              // tile 暂存，按 uint 平铺访问：第 j 个 = s_v[j >> 2][j & 3]
shared uint s_sum[WG + WG / 32u];  // 段和，每 32 个后空一位避免 raking 时 bank 冲突
shared uint s_blk[33];             // 32 个块和（扫描后为块前缀）+ tile 总和

void main() {
	uint t = gl_LocalInvocationIndex;
	uint n = uint(max(num_cells, 0));
	uint ts = t + (t >> 5u);
	uint carry = 0u;  // 之前各 tile 的总和
	for (uint base = 0u; base < n; base += TILE) {
		// 1. 合并读入本 tile
		for (uint k = 0u; k < 4u; k++) {
			uint j = k * WG + t;
			s_v[j >> 2u][j & 3u] = base + j < n ? cell_count[base + j] : 0u;
		}
		barrier();

		// 2. 本线程 4 格：段内 exclusive + 段和
		uvec4 v = s_v[t];
		uvec4 ex = uvec4(0u, v.x, v.x + v.y, v.x + v.y + v.z);
		s_sum[ts] = ex.w + v.w;
		barrier();

		// 3. 1024 个段和做 exclusive scan：32 线程各串行扫 32 个（raking），线程 0 再扫 32 个块和
		if (t < 32u) {
			uint x[32];
			for (uint j = 0u; j < 32u; j++) x[j] = s_sum[t * 33u + j];
			uint run = 0u;
			for (uint j = 0u; j < 32u; j++) {
				s_sum[t * 33u + j] = run;
				run += x[j];
			}
			s_blk[t] = run;
		}
		barrier();
		if (t == 0u) {
			uint x[32];
			for (uint j = 0u; j < 32u; j++) x[j] = s_blk[j];
			uint run = 0u;
			for (uint j = 0u; j < 32u; j++) {
				s_blk[j] = run;
				run += x[j];
			}
			s_blk[32] = run;
		}
		barrier();

		// 4. 结果写回 shared，再合并写出
		s_v[t] = ex + (carry + s_blk[t >> 5u] + s_sum[ts]);
		carry += s_blk[32];
		barrier();
		for (uint k = 0u; k < 4u; k++) {
			uint j = k * WG + t;
			if (base + j < n) {
				uint r = s_v[j >> 2u][j & 3u];
				cell_start[base + j]  = r;
				cell_offset[base + j] = r;
			}
		}
		barrier();
	}
}
