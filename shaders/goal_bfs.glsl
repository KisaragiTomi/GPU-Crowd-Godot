#[compute]
#version 450

// 16x16：mode 1 每个工作组负责一个 tile
// 每帧（最近敌人）：0 init → 1 松弛 → 2 梯度
// 开局目标场（GPUSWEField.build_all_goal_fields）：3 init →（CPU 写种子）→ 1 松弛 → 4 最大距离 → 5 写出全部组
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint  faction_presence[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float terrain[];          };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { uint  fac_to_group[];     };
layout(set = 0, binding = 3, std430) restrict           buffer B3 { uint  bfs_dist[];         };
layout(set = 0, binding = 4, std430) restrict writeonly buffer B4 { float goal_dist_all_out[];};
layout(set = 0, binding = 5, std430) restrict writeonly buffer B5 { float goal_dist_out[];    };
layout(set = 0, binding = 6, std430) restrict writeonly buffer B6 { float gdir_x_out[];       };
layout(set = 0, binding = 7, std430) restrict writeonly buffer B7 { float gdir_y_out[];       };
layout(set = 0, binding = 8, std430) restrict           buffer B8 { uint  bfs_changed[];      };  // [0..2] 全局变化标志，3 槽轮换；[3] 开局目标场的最大可达距离
layout(set = 0, binding = 9, std430) restrict readonly  buffer B9 { float dir_lut[];          };  // 开局目标场的梯度方向表（GPUSWEField._ensure_dir_lut）
layout(set = 0, binding = 10, std430) restrict readonly buffer B10 { float ground_cost[]; };
layout(set = 0, binding = 11, std430) restrict buffer B11 { uint bfs_cost[]; }; // weighted movement cost; bfs_dist remains cell distance

layout(push_constant, std430) uniform Params {
	int field_gw;
	int field_gh;
	int mode;
	int target_group;
	int num_factions;
	int relax_pass;
	int lut_cap;      // mode 5：方向表覆盖的 v 范围
	int num_groups;   // mode 5：写入的组数
};

const uint BFS_WALL      = 0xFFFFFFFFu;
const uint BFS_UNREACHED = 0xFFFFFFFEu;
const int  TILE          = 16;
const int  HALO_W        = TILE + 2;      // tile + 1 格 halo
const int  HALO_N        = HALO_W * HALO_W;
const int  TILE_ITERS    = TILE * TILE;   // tile 内迭代上限 = tile 面积，保证 tile 内收敛

shared uint s_dist[2][HALO_N];   // Jacobi 双缓冲
shared uint s_cost[2][HALO_N];   // same wavefront, weighted by ground difficulty
shared uint s_changed[3];        // 轮换标志：本轮写 [it%3]，同时清零 [(it+1)%3]

// RELAX：tile + halo 读进 shared，在 tile 内反复 4 邻居松弛直到不再变化，再写回
// 所有线程（含越界线程）执行相同次数的 barrier：只允许整组一致的提前 return
void relax_tile() {
	int lx = int(gl_LocalInvocationID.x);
	int ly = int(gl_LocalInvocationID.y);
	int li = int(gl_LocalInvocationIndex);
	int ox = int(gl_WorkGroupID.x) * TILE - 1;   // halo 左上角的全局坐标
	int oy = int(gl_WorkGroupID.y) * TILE - 1;

	// 全局 early-out：第 p 次写 [p%3]、清 [(p+1)%3]、读上一次的 [(p+2)%3]；上一次全图无变化即已到不动点
	if (li == 0 && gl_WorkGroupID.x == 0u && gl_WorkGroupID.y == 0u) bfs_changed[(relax_pass + 1) % 3] = 0u;
	if (relax_pass > 0 && bfs_changed[(relax_pass + 2) % 3] == 0u) return;

	for (int h = li; h < HALO_N; h += TILE * TILE) {
		int gx = ox + h % HALO_W;
		int gy = oy + h / HALO_W;
		uint v = BFS_WALL;   // 越界当墙
		uint c = BFS_WALL;
		if (gx >= 0 && gx < field_gw && gy >= 0 && gy < field_gh) {
			v = bfs_dist[gy * field_gw + gx];
			c = bfs_cost[gy * field_gw + gx];
		}
		s_dist[0][h] = v;
		s_dist[1][h] = v;
		s_cost[0][h] = c;
		s_cost[1][h] = c;
	}
	if (li < 3) s_changed[li] = 0u;
	barrier();

	int si = (ly + 1) * HALO_W + (lx + 1);
	uint orig = s_dist[0][si];
	uint cur = orig;
	uint orig_cost = s_cost[0][si];
	uint cur_cost = orig_cost;
	bool can_relax = cur != BFS_WALL && cur != 0u;   // 墙（含越界）和种子永不改变
	uint step_cost = 1u;
	if (can_relax) step_cost = uint(clamp(round(ground_cost[(oy + 1 + ly) * field_gw + ox + 1 + lx]), 1.0, 4.0));
	int src = 0;
	for (int it = 0; it < TILE_ITERS; it++) {
		if (can_relax) {
			uint nd = min(min(s_dist[src][si - 1],      s_dist[src][si + 1]),
			              min(s_dist[src][si - HALO_W], s_dist[src][si + HALO_W]));
			if (nd < BFS_UNREACHED && nd + 1u < cur) {
				cur = nd + 1u;
				s_changed[it % 3] = 1u;
			}
			s_dist[1 - src][si] = cur;
			uint nc = min(min(s_cost[src][si - 1], s_cost[src][si + 1]),
			              min(s_cost[src][si - HALO_W], s_cost[src][si + HALO_W]));
			// Guard before adding so wall/unreachable sentinels cannot wrap to a cheap route.
			if (nc < BFS_UNREACHED - step_cost && nc + step_cost < cur_cost) {
				cur_cost = nc + step_cost;
				s_changed[it % 3] = 1u;
			}
			s_cost[1 - src][si] = cur_cost;
		}
		// [(it+1)%3] 即 [(it-2)%3]，上次在 barrier(it-2) 之后被读；此时全组已越过 barrier(it-1)，可安全清零
		if (li == 0) s_changed[(it + 1) % 3] = 0u;
		barrier();
		if (s_changed[it % 3] == 0u) break;   // 全组读同一个值，分支一致
		src = 1 - src;
	}

	if (cur < orig || cur_cost < orig_cost) {
		atomicMin(bfs_dist[(oy + 1 + ly) * field_gw + (ox + 1 + lx)], cur);
		atomicMin(bfs_cost[(oy + 1 + ly) * field_gw + (ox + 1 + lx)], cur_cost);
		bfs_changed[relax_pass % 3] = 1u;
	}
}

// ── 开局目标场（mode 3..5）：普通地面保留原 CPU 版结果 ──

shared uint s_max;

// mode 4：可达格的最大距离 → bfs_changed[3]。全组都执行 barrier，越界线程只是不参与比较
void reduce_max() {
	if (gl_LocalInvocationIndex == 0u) s_max = 0u;
	barrier();
	int fx = int(gl_GlobalInvocationID.x);
	int fy = int(gl_GlobalInvocationID.y);
	if (fx < field_gw && fy < field_gh) {
		uint d = bfs_cost[fy * field_gw + fx];
		if (d < BFS_UNREACHED) atomicMax(s_max, d);
	}
	barrier();
	if (gl_LocalInvocationIndex == 0u && s_max > 0u) atomicMax(bfs_changed[3], s_max);
}

// 开局移动代价：可达格取加权距离，墙和未到达格取最大可达代价 + 1。
int start_dist(int gi, int fill_d) {
	uint u = bfs_cost[gi];
	return u < BFS_UNREACHED ? int(u) : fill_d;
}

// 按 (a, b) = (|ddx|, |ddy|) 查原 CPU 版 |ddx| / sqrt(ddx² + ddy²)、|ddy| / sqrt(...) 的 float32 结果。
// 普通地面相邻距离差 ≤ 1；加权地面的其它方向走下面的直接计算分支。
// 表中每个 v 占 12 个 float：(v, s) 与 (v, v + s) 两组，s = 0..2，每组存两个分量
vec2 dir_magnitude(int a, int b) {
	int hi = max(a, b);
	int lo = min(a, b);
	vec2 m;
	if (lo <= 2 && hi < lut_cap) {
		int k = hi * 12 + lo * 2;
		m = vec2(dir_lut[k], dir_lut[k + 1]);
	} else if (hi - lo <= 2 && lo < lut_cap) {
		int k = lo * 12 + 6 + (hi - lo) * 2;
		m = vec2(dir_lut[k + 1], dir_lut[k]);
	} else {
		// 加权地面可能落在表外，直接归一化。
		float l = sqrt(float(hi) * float(hi) + float(lo) * float(lo));
		m = vec2(float(hi) / l, float(lo) / l);
	}
	return a >= b ? m : m.yx;
}

// -sign(dd) * mag（mag ≥ 0）：直接置符号位，dd = 0 时得到 -0.0，与原 CPU 版 -ddx / l 相同
float neg_by_sign(int dd, float mag) {
	return uintBitsToFloat(floatBitsToUint(mag) | (dd >= 0 ? 0x80000000u : 0u));
}

// Cost discontinuities need downhill differences: a central difference can point
// into more expensive ground or cancel between two equally cheap detours.
// Keep the original gradient where the weighted and ordinary neighborhoods agree.
vec2 ground_direction(int fx, int fy, int gi, vec2 original_dir) {
	uint d = bfs_cost[gi];
	if (d >= BFS_UNREACHED) return original_dir;
	int il = fx > 0 ? gi - 1 : gi;
	int ir = fx < field_gw - 1 ? gi + 1 : gi;
	int iu = fy > 0 ? gi - field_gw : gi;
	int id = fy < field_gh - 1 ? gi + field_gw : gi;
	uint dl = bfs_cost[il], dr = bfs_cost[ir], du = bfs_cost[iu], dd = bfs_cost[id];
	if (d == bfs_dist[gi] && dl == bfs_dist[il] && dr == bfs_dist[ir]
		&& du == bfs_dist[iu] && dd == bfs_dist[id]) return original_dir;
	if (d == 0u) return vec2(0.0);
	float x = dl <= dr ? -float(d - min(dl, d)) : float(d - min(dr, d));
	float y = du <= dd ? -float(d - min(du, d)) : float(d - min(dd, d));
	vec2 downhill = vec2(x, y);
	float len = length(downhill);
	return len > 0.0 ? downhill / len : vec2(0.0);
}

// mode 5：开局目标场的距离与梯度，写入全部 num_groups 组
void write_start_field(int fx, int fy, int gi) {
	int fill_d = int(bfs_changed[3]) + 1;
	int d = start_dist(gi, fill_d);
	float gx = 0.0;
	float gy = 0.0;
	if (terrain[gi] <= 0.5) {
		// 差分规则同原 CPU 版：内部中心差分，边界单侧差分
		int ddx = 0;
		int ddy = 0;
		if (fx > 0 && fx < field_gw - 1) ddx = start_dist(gi + 1, fill_d) - start_dist(gi - 1, fill_d);
		else if (fx == 0 && field_gw > 1) ddx = start_dist(gi + 1, fill_d) - d;
		else if (fx == field_gw - 1 && field_gw > 1) ddx = d - start_dist(gi - 1, fill_d);
		if (fy > 0 && fy < field_gh - 1) ddy = start_dist(gi + field_gw, fill_d) - start_dist(gi - field_gw, fill_d);
		else if (fy == 0 && field_gh > 1) ddy = start_dist(gi + field_gw, fill_d) - d;
		else if (fy == field_gh - 1 && field_gh > 1) ddy = d - start_dist(gi - field_gw, fill_d);
		if (ddx != 0 || ddy != 0) {
			vec2 m = dir_magnitude(abs(ddx), abs(ddy));
			gx = neg_by_sign(ddx, m.x);
			gy = neg_by_sign(ddy, m.y);
		}
	}
	vec2 dir = ground_direction(fx, fy, gi, vec2(gx, gy));
	goal_dist_out[gi] = float(d);
	int cells = field_gw * field_gh;
	for (int g = 0; g < num_groups; g++) {
		int o = g * cells + gi;
		goal_dist_all_out[o] = bfs_dist[gi] < BFS_UNREACHED ? float(bfs_dist[gi]) : float(fill_d);
		gdir_x_out[o] = dir.x;
		gdir_y_out[o] = dir.y;
	}
}

void main() {
	if (mode == 1) {
		relax_tile();
		return;
	}
	if (mode == 4) {
		reduce_max();
		return;
	}

	int fx = int(gl_GlobalInvocationID.x);
	int fy = int(gl_GlobalInvocationID.y);
	if (fx >= field_gw || fy >= field_gh) return;
	int gi = fy * field_gw + fx;

	if (mode == 0) {
		if (gi == 0) { bfs_changed[0] = 0u; bfs_changed[1] = 0u; bfs_changed[2] = 0u; }
		// INIT: seed = enemy faction cells, wall = MAX
		if (terrain[gi] > 0.5) {
			bfs_dist[gi] = 0xFFFFFFFFu;
			bfs_cost[gi] = BFS_WALL;
			return;
		}
		uint fp = faction_presence[gi];
		bool has_enemy = false;
		for (int f = 0; f < num_factions; f++) {
			if ((fp & (1u << uint(f))) != 0u && fac_to_group[f] != uint(target_group)) {
				has_enemy = true;
				break;
			}
		}
		bfs_dist[gi] = has_enemy ? 0u : 0xFFFFFFFEu;
		bfs_cost[gi] = has_enemy ? 0u : BFS_UNREACHED;
	}
	else if (mode == 2) {
		// GRADIENT: write float distance + gradient to output buffers
		int off = target_group * field_gw * field_gh;
		uint ud = bfs_cost[gi];
		float d = (ud >= 0xFFFFFFFEu) ? 1e6 : float(ud);
		goal_dist_all_out[off + gi] = bfs_dist[gi] >= BFS_UNREACHED ? 1e6 : float(bfs_dist[gi]);
		if (target_group == 0) goal_dist_out[gi] = d;

		if (terrain[gi] > 0.5 || ud >= 0xFFFFFFFEu) {
			gdir_x_out[off + gi] = 0.0;
			gdir_y_out[off + gi] = 0.0;
			return;
		}

		float dl = (fx > 0)            ? float(min(bfs_cost[gi - 1],         0xFFFFFFFEu)) : d;
		float dr = (fx < field_gw - 1) ? float(min(bfs_cost[gi + 1],         0xFFFFFFFEu)) : d;
		float du = (fy > 0)            ? float(min(bfs_cost[gi - field_gw],   0xFFFFFFFEu)) : d;
		float dd = (fy < field_gh - 1) ? float(min(bfs_cost[gi + field_gw],   0xFFFFFFFEu)) : d;

		float ddx = dr - dl;
		float ddy = dd - du;
		float l = sqrt(ddx * ddx + ddy * ddy);
		vec2 dir = ground_direction(fx, fy, gi, l > 1e-6 ? vec2(-ddx / l, -ddy / l) : vec2(0.0));
		gdir_x_out[off + gi] = dir.x;
		gdir_y_out[off + gi] = dir.y;
	}
	else if (mode == 3) {
		// 开局 INIT：墙 = MAX，其余未到达（种子随后由 CPU 写 0）；清空全部标志和最大距离
		if (gi == 0) { bfs_changed[0] = 0u; bfs_changed[1] = 0u; bfs_changed[2] = 0u; bfs_changed[3] = 0u; }
		bfs_dist[gi] = terrain[gi] > 0.5 ? BFS_WALL : BFS_UNREACHED;
		bfs_cost[gi] = terrain[gi] > 0.5 ? BFS_WALL : BFS_UNREACHED;
	}
	else if (mode == 5) {
		write_start_field(fx, fy, gi);
	}
}
