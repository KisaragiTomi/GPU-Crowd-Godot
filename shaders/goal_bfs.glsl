#[compute]
#version 450

// 16x16：mode 1 每个工作组负责一个 tile
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint  faction_presence[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float terrain[];          };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { uint  fac_to_group[];     };
layout(set = 0, binding = 3, std430) restrict           buffer B3 { uint  bfs_dist[];         };
layout(set = 0, binding = 4, std430) restrict writeonly buffer B4 { float goal_dist_all_out[];};
layout(set = 0, binding = 5, std430) restrict writeonly buffer B5 { float goal_dist_out[];    };
layout(set = 0, binding = 6, std430) restrict writeonly buffer B6 { float gdir_x_out[];       };
layout(set = 0, binding = 7, std430) restrict writeonly buffer B7 { float gdir_y_out[];       };
layout(set = 0, binding = 8, std430) restrict           buffer B8 { uint  bfs_changed[];      };  // 全局变化标志，3 槽轮换

layout(push_constant, std430) uniform Params {
	int field_gw;
	int field_gh;
	int mode;
	int target_group;
	int num_factions;
	int relax_pass;
	int _pad1;
	int _pad2;
};

const uint BFS_WALL      = 0xFFFFFFFFu;
const uint BFS_UNREACHED = 0xFFFFFFFEu;
const int  TILE          = 16;
const int  HALO_W        = TILE + 2;      // tile + 1 格 halo
const int  HALO_N        = HALO_W * HALO_W;
const int  TILE_ITERS    = TILE * TILE;   // tile 内迭代上限 = tile 面积，保证 tile 内收敛

shared uint s_dist[2][HALO_N];   // Jacobi 双缓冲
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
		if (gx >= 0 && gx < field_gw && gy >= 0 && gy < field_gh) v = bfs_dist[gy * field_gw + gx];
		s_dist[0][h] = v;
		s_dist[1][h] = v;
	}
	if (li < 3) s_changed[li] = 0u;
	barrier();

	int si = (ly + 1) * HALO_W + (lx + 1);
	uint orig = s_dist[0][si];
	uint cur = orig;
	bool can_relax = cur != BFS_WALL && cur != 0u;   // 墙（含越界）和种子永不改变
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
		}
		// [(it+1)%3] 即 [(it-2)%3]，上次在 barrier(it-2) 之后被读；此时全组已越过 barrier(it-1)，可安全清零
		if (li == 0) s_changed[(it + 1) % 3] = 0u;
		barrier();
		if (s_changed[it % 3] == 0u) break;   // 全组读同一个值，分支一致
		src = 1 - src;
	}

	if (cur < orig) {
		atomicMin(bfs_dist[(oy + 1 + ly) * field_gw + (ox + 1 + lx)], cur);
		bfs_changed[relax_pass % 3] = 1u;
	}
}

void main() {
	if (mode == 1) {
		relax_tile();
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
	}
	else if (mode == 2) {
		// GRADIENT: write float distance + gradient to output buffers
		int off = target_group * field_gw * field_gh;
		uint ud = bfs_dist[gi];
		float d = (ud >= 0xFFFFFFFEu) ? 1e6 : float(ud);
		goal_dist_all_out[off + gi] = d;
		if (target_group == 0) goal_dist_out[gi] = d;

		if (terrain[gi] > 0.5 || ud >= 0xFFFFFFFEu) {
			gdir_x_out[off + gi] = 0.0;
			gdir_y_out[off + gi] = 0.0;
			return;
		}

		float dl = (fx > 0)            ? float(min(bfs_dist[gi - 1],         0xFFFFFFFEu)) : d;
		float dr = (fx < field_gw - 1) ? float(min(bfs_dist[gi + 1],         0xFFFFFFFEu)) : d;
		float du = (fy > 0)            ? float(min(bfs_dist[gi - field_gw],   0xFFFFFFFEu)) : d;
		float dd = (fy < field_gh - 1) ? float(min(bfs_dist[gi + field_gw],   0xFFFFFFFEu)) : d;

		float ddx = dr - dl;
		float ddy = dd - du;
		float l = sqrt(ddx * ddx + ddy * ddy);
		if (l > 1e-6) {
			gdir_x_out[off + gi] = -ddx / l;
			gdir_y_out[off + gi] = -ddy / l;
		} else {
			gdir_x_out[off + gi] = 0.0;
			gdir_y_out[off + gi] = 0.0;
		}
	}
}
