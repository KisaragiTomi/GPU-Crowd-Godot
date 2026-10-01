#[compute]
#version 450

// 选中 agent 的 GPU 查询，按 mode 分成几个小 pass（之间需要 barrier）：
//   0 清哨兵（1 线程） → 1 最近敌人距离²（全体 agent） → 2 最近敌人下标（全体 agent） → 3 汇总（1 线程）
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer B0 { float pos_x[];         };
layout(set = 0, binding = 1, std430) restrict readonly buffer B1 { float pos_y[];         };
layout(set = 0, binding = 2, std430) restrict readonly buffer B2 { uint  agent_info[];    };
layout(set = 0, binding = 3, std430) restrict readonly buffer B3 { int   damage_acc[];    };
layout(set = 0, binding = 4, std430) restrict readonly buffer B4 { uint  alliance[];      };
layout(set = 0, binding = 5, std430) restrict readonly buffer B5 { uint  cell_attacker[]; };
layout(set = 0, binding = 6, std430) restrict          buffer B6 { uint  q[];             };

layout(push_constant, std430) uniform Params {
	int   agent_count;  // 0
	int   sel;          // 4
	int   mode;         // 8
	int   field_gw;     // 12
	int   field_gh;     // 16
	float field_cs;     // 20
	int   _p0;          // 24
	int   _p1;          // 28
};

// 输出布局（与 gpu_agents.gd 的 take_select_inspect 对应）
const uint Q_MIN_DSQ  = 0u;   // 最近敌人距离² 的位模式（非负 float 可直接按 uint 比较）
const uint Q_NE       = 1u;   // 最近敌人下标
const uint Q_SEL_INFO = 2u;
const uint Q_SEL_DMG  = 3u;
const uint Q_SEL_X    = 4u;
const uint Q_SEL_Y    = 5u;
const uint Q_NE_INFO  = 6u;
const uint Q_NE_X     = 7u;
const uint Q_NE_Y     = 8u;
const uint Q_CELL     = 9u;   // 选中 agent 所在 field 格
const uint Q_BLK      = 10u;  // 该格的 cell_attacker（blocker）
const uint Q_BLK_X    = 11u;
const uint Q_BLK_Y    = 12u;
const uint NONE       = 0xFFFFFFFFu;

shared uint s_min[64];

// j 是选中 agent 的存活敌人时返回距离² 的位模式，否则 NONE
uint enemy_dsq_bits(uint j) {
	if (int(j) >= agent_count || int(j) == sel) return NONE;
	uint ji = agent_info[j];
	if ((ji & 1u) == 0u) return NONE;
	uint s = uint(sel);
	uint ally = alliance[(agent_info[s] >> 1u) & 0x1Fu];
	if (((ally >> ((ji >> 1u) & 0x1Fu)) & 1u) != 0u) return NONE;
	// 与 CPU 的 Vector2.distance_squared_to 同序同精度：precise 禁止合并成 FMA
	precise float dx = pos_x[s] - pos_x[j];
	precise float dy = pos_y[s] - pos_y[j];
	precise float dsq = dx * dx + dy * dy;
	return floatBitsToUint(dsq);
}

void main() {
	uint j = gl_GlobalInvocationID.x;
	uint lid = gl_LocalInvocationID.x;

	if (mode == 0) {
		if (j == 0u) {
			q[Q_MIN_DSQ] = NONE;
			q[Q_NE] = NONE;
		}
		return;
	}

	if (mode == 1) {
		// 组内先归约，每个工作组只做一次全局 atomicMin
		s_min[lid] = enemy_dsq_bits(j);
		barrier();
		for (uint w = 32u; w > 0u; w >>= 1u) {
			if (lid < w) s_min[lid] = min(s_min[lid], s_min[lid + w]);
			barrier();
		}
		if (lid == 0u && s_min[0] != NONE) atomicMin(q[Q_MIN_DSQ], s_min[0]);
		return;
	}

	if (mode == 2) {
		// 距离² 等于最小值的敌人里取最小下标（与 CPU 严格小于的平局规则一致）
		uint d = enemy_dsq_bits(j);
		if (d != NONE && d == q[Q_MIN_DSQ]) atomicMin(q[Q_NE], j);
		return;
	}

	// mode 3：单线程汇总
	if (j != 0u) return;
	uint s = uint(sel);
	float sx = pos_x[s];
	float sy = pos_y[s];
	q[Q_SEL_INFO] = agent_info[s];
	q[Q_SEL_DMG]  = uint(damage_acc[s]);
	q[Q_SEL_X]    = floatBitsToUint(sx);
	q[Q_SEL_Y]    = floatBitsToUint(sy);

	uint ne = q[Q_NE];
	bool has_ne = ne != NONE;
	q[Q_NE_INFO] = has_ne ? agent_info[ne] : 0u;
	q[Q_NE_X]    = has_ne ? floatBitsToUint(pos_x[ne]) : 0u;
	q[Q_NE_Y]    = has_ne ? floatBitsToUint(pos_y[ne]) : 0u;

	int cx = clamp(int(sx / field_cs), 0, field_gw - 1);
	int cy = clamp(int(sy / field_cs), 0, field_gh - 1);
	uint ci = uint(cy * field_gw + cx);
	uint b = cell_attacker[ci];
	bool has_b = b < uint(agent_count);
	q[Q_CELL]  = ci;
	q[Q_BLK]   = b;
	q[Q_BLK_X] = has_b ? floatBitsToUint(pos_x[b]) : 0u;
	q[Q_BLK_Y] = has_b ? floatBitsToUint(pos_y[b]) : 0u;
}
