// Resolve diagonal crossings in the similarity graph.
// Each thread processes one 2x2 pixel block.
// Three heuristics vote to decide which diagonal to keep:
//   1. Curve length -- longer connected chain wins
//   2. Sparse pixels -- smaller component (BFS in 8x8 window) wins
//   3. Islands -- valence-1 endpoint gets fixed weight 5 (early return on first found)
// Fully-connected blocks (all 4 cardinal edges) remove both diagonals.
// Single diagonals in fully-connected blocks are also removed.
// Ties remove both diagonals (per paper).

// Read from a snapshot of the graph (pre-resolution state) so all threads
// see consistent data regardless of execution order. Write to the live graph.
struct Uniforms {
    uint img_w;
    uint img_h;
    uint graph_stride; // 2*img_w + 1
    uint _pad;
};
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
ConstantBuffer<Uniforms> u : register(b0, space2);
#else
ConstantBuffer<Uniforms> u;
#endif

[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> graph_in : register(t0, space0);
#else
StructuredBuffer<uint> graph_in;
#endif
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
RWStructuredBuffer<uint> graph_out : register(u0, space1);
#else
RWStructuredBuffer<uint> graph_out;
#endif

uint g(uint gx, uint gy) { return graph_in[gy * u.graph_stride + gx]; }
void gs(uint gx, uint gy, uint val) { graph_out[gy * u.graph_stride + gx] = val; }

// Direction bitmask encoding (matches reference)
static const uint DIR_NW = 1u;
static const uint DIR_W  = 2u;
static const uint DIR_SW = 4u;
static const uint DIR_S  = 8u;
static const uint DIR_SE = 16u;
static const uint DIR_E  = 32u;
static const uint DIR_NE = 64u;
static const uint DIR_N  = 128u;

// Compute 8-connected valence bitmask for pixel (x,y)
uint valence_mask(uint x, uint y) {
    uint v = 0u;
    uint gx = 2u * x + 1u;
    uint gy = 2u * y + 1u;
    uint gw = u.graph_stride;
    uint gh = 2u * u.img_h + 1u;

    // Cardinal
    if (gx >= 2u && g(gx - 1u, gy) != 0u) v |= DIR_W;
    if (gx + 1u < gw && g(gx + 1u, gy) != 0u) v |= DIR_E;
    if (gy >= 2u && g(gx, gy - 1u) != 0u) v |= DIR_N;
    if (gy + 1u < gh && g(gx, gy + 1u) != 0u) v |= DIR_S;
    // Diagonal
    if (gx >= 2u && gy >= 2u && (g(gx - 1u, gy - 1u) & 1u) != 0u) v |= DIR_NW;
    if (gx + 1u < gw && gy >= 2u && (g(gx + 1u, gy - 1u) & 2u) != 0u) v |= DIR_NE;
    if (gx >= 2u && gy + 1u < gh && (g(gx - 1u, gy + 1u) & 2u) != 0u) v |= DIR_SW;
    if (gx + 1u < gw && gy + 1u < gh && (g(gx + 1u, gy + 1u) & 1u) != 0u) v |= DIR_SE;
    return v;
}

// Count bits in valence mask
uint valence_count(uint mask) {
    return uint(countbits(mask));
}

// Walk a chain of valence-2 nodes using XOR direction tracking.
uint walk_chain(uint sx, uint sy, uint fx, uint fy) {
    uint cx = sx, cy = sy;
    uint length = 1u;

    int idx = int(fx) - int(sx);
    int idy = int(fy) - int(sy);
    uint pred_dir = 0u;
    if (idx ==  1 && idy ==  0) pred_dir = DIR_E;
    if (idx == -1 && idy ==  0) pred_dir = DIR_W;
    if (idx ==  0 && idy ==  1) pred_dir = DIR_S;
    if (idx ==  0 && idy == -1) pred_dir = DIR_N;
    if (idx ==  1 && idy ==  1) pred_dir = DIR_SE;
    if (idx == -1 && idy == -1) pred_dir = DIR_NW;
    if (idx ==  1 && idy == -1) pred_dir = DIR_NE;
    if (idx == -1 && idy ==  1) pred_dir = DIR_SW;

    for (uint iter = 0u; iter < 200u; iter++) {
        uint mask = valence_mask(cx, cy);
        if (valence_count(mask) != 2u) break;

        uint next_dir = mask ^ pred_dir;

        int dx = 0, dy = 0;
        uint reverse = 0u;
        switch (next_dir) {
            case DIR_N:  dy = -1; reverse = DIR_S;  break;
            case DIR_NE: dx =  1; dy = -1; reverse = DIR_SW; break;
            case DIR_E:  dx =  1; reverse = DIR_W;  break;
            case DIR_SE: dx =  1; dy =  1; reverse = DIR_NW; break;
            case DIR_S:  dy =  1; reverse = DIR_N;  break;
            case DIR_SW: dx = -1; dy =  1; reverse = DIR_NE; break;
            case DIR_W:  dx = -1; reverse = DIR_E;  break;
            case DIR_NW: dx = -1; dy = -1; reverse = DIR_SE; break;
            default: return length;
        }

        cx = uint(int(cx) + dx);
        cy = uint(int(cy) + dy);
        pred_dir = reverse;
        length++;
    }
    return length;
}

// Pack 64 labels (each 0/1/2, 2 bits) into a uint4 (128 bits). Keeps the
// labels array in registers instead of spilling 256 bytes of int[64] to
// scratch. idx = row * 8 + col; uint_idx = idx / 16; shift = (idx % 16) * 2.
uint get_label(uint4 labels, int row, int col) {
    int idx = row * 8 + col;
    int uint_idx = idx >> 4;
    int shift = (idx & 15) << 1;
    uint v;
    if (uint_idx == 0)      v = labels.x;
    else if (uint_idx == 1) v = labels.y;
    else if (uint_idx == 2) v = labels.z;
    else                    v = labels.w;
    return (v >> shift) & 3u;
}

void set_label(inout uint4 labels, int row, int col, uint value) {
    int idx = row * 8 + col;
    int uint_idx = idx >> 4;
    int shift = (idx & 15) << 1;
    uint clear = ~(3u << shift);
    uint v = (value & 3u) << shift;
    if (uint_idx == 0)      labels.x = (labels.x & clear) | v;
    else if (uint_idx == 1) labels.y = (labels.y & clear) | v;
    else if (uint_idx == 2) labels.z = (labels.z & clear) | v;
    else                    labels.w = (labels.w & clear) | v;
}

// Simultaneous A/B component labeling in 8x8 window around the crossing.
void sparse_pixel_sizes(uint bx, uint by, out uint sizeA, out uint sizeB) {
    // 8x8 labels packed into uint4. Seeds at (col,row) ∈ {(3,3),(4,4)}=1
    // (main/backslash) and {(4,3),(3,4)}=2 (anti/slash):
    //   idx 27 (row 3, col 3) → uint_idx 1, shift 22 → 1u << 22 in .y
    //   idx 28 (row 3, col 4) → uint_idx 1, shift 24 → 2u << 24 in .y
    //   idx 35 (row 4, col 3) → uint_idx 2, shift  6 → 2u <<  6 in .z
    //   idx 36 (row 4, col 4) → uint_idx 2, shift  8 → 1u <<  8 in .z
    uint4 labels = uint4(
        0u,
        (1u << 22) | (2u << 24),
        (2u <<  6) | (1u <<  8),
        0u
    );

    sizeA = 0u;
    sizeB = 0u;

    // BFS frontier: bit i set = cell idx i (= row*8+col) needs processing.
    // 64 bits = uint2, lives in registers. Each cell enters/leaves the
    // frontier exactly once — same total work as a queue, but no scratch.
    //   bit 27 (row 3, col 3) and bit 28 (row 3, col 4) → frontier.x
    //   bit 35 (row 4, col 3) and bit 36 (row 4, col 4) → frontier.y at
    //                                                     bit 3 / 4.
    uint2 frontier = uint2(
        (1u << 27) | (1u << 28),
        (1u <<  3) | (1u <<  4)
    );

    static const int dcol[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
    static const int drow[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
    static const uint dirs[8] = { DIR_NW, DIR_W, DIR_SW, DIR_S, DIR_SE, DIR_E, DIR_NE, DIR_N };

    while (frontier.x != 0u || frontier.y != 0u) {
        // Pop the lowest set bit from frontier.
        int idx;
        if (frontier.x != 0u) {
            idx = firstbitlow(frontier.x);
            frontier.x &= ~(1u << uint(idx));
        } else {
            int hi = firstbitlow(frontier.y);
            frontier.y &= ~(1u << uint(hi));
            idx = 32 + hi;
        }

        int row = idx >> 3;
        int col = idx & 7;
        int label = int(get_label(labels, row, col));

        int px = int(bx) + col - 3;
        int py = int(by) + row - 3;
        if (px < 0 || py < 0 || px >= int(u.img_w) || py >= int(u.img_h)) continue;

        uint vmask = valence_mask(uint(px), uint(py));

        for (uint d = 0u; d < 8u; d++) {
            if ((vmask & dirs[d]) == 0u) continue;
            int nc = col + dcol[d];
            int nr = row + drow[d];
            if (nc < 0 || nc >= 8 || nr < 0 || nr >= 8) continue;
            if (get_label(labels, nr, nc) != 0u) continue;

            int npx = int(bx) + nc - 3;
            int npy = int(by) + nr - 3;
            if (npx < 0 || npy < 0 || npx >= int(u.img_w) || npy >= int(u.img_h)) continue;

            set_label(labels, nr, nc, uint(label));
            if (label == 1) sizeA++;
            else sizeB++;

            // Set the new cell's bit in the frontier.
            int n_idx = nr * 8 + nc;
            if (n_idx < 32) frontier.x |= 1u << uint(n_idx);
            else            frontier.y |= 1u << uint(n_idx - 32);
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint bx = tid.x;
    uint by = tid.y;
    if (bx >= u.img_w - 1u || by >= u.img_h - 1u) return;

    uint corner_gx = 2u * bx + 2u;
    uint corner_gy = 2u * by + 2u;
    uint flags = g(corner_gx, corner_gy);

    if (flags == 0u) return;

    // Fully connected check: all 4 cardinal edges present
    uint top = g(corner_gx, corner_gy - 1u);
    uint bottom = g(corner_gx, corner_gy + 1u);
    uint left = g(corner_gx - 1u, corner_gy);
    uint right = g(corner_gx + 1u, corner_gy);
    bool fully_connected = top != 0u && bottom != 0u && left != 0u && right != 0u;

    // Single diagonal in fully-connected block: remove it
    if (flags == 1u || flags == 2u) {
        if (fully_connected) {
            gs(corner_gx, corner_gy, 0u);
        }
        return;
    }

    // flags == 3u: crossing (both diagonals)
    if (fully_connected) {
        gs(corner_gx, corner_gy, 0u);
        return;
    }

    int main_vote = 0;
    int anti_vote = 0;

    // Heuristic 1: Curve length
    uint main_len = walk_chain(bx, by, bx + 1u, by + 1u)
                  + walk_chain(bx + 1u, by + 1u, bx, by);
    uint anti_len = walk_chain(bx + 1u, by, bx, by + 1u)
                  + walk_chain(bx, by + 1u, bx + 1u, by);
    if (main_len > anti_len) main_vote += int(main_len - anti_len);
    else if (anti_len > main_len) anti_vote += int(anti_len - main_len);

    // Heuristic 2: Sparse pixels
    uint main_size, anti_size;
    sparse_pixel_sizes(bx, by, main_size, anti_size);
    if (main_size < anti_size) main_vote += int(anti_size - main_size);
    else if (anti_size < main_size) anti_vote += int(main_size - anti_size);

    // Heuristic 3: Islands (valence-1 endpoint)
    uint v_main0 = valence_count(valence_mask(bx, by));
    uint v_main1 = valence_count(valence_mask(bx + 1u, by + 1u));
    uint v_anti0 = valence_count(valence_mask(bx + 1u, by));
    uint v_anti1 = valence_count(valence_mask(bx, by + 1u));

    if (v_main0 == 1u) main_vote += 5;
    else if (v_main1 == 1u) main_vote += 5;
    else if (v_anti0 == 1u) anti_vote += 5;
    else if (v_anti1 == 1u) anti_vote += 5;

    // Resolve
    if (main_vote > anti_vote) {
        gs(corner_gx, corner_gy, 1u); // keep main (backslash)
    } else if (anti_vote > main_vote) {
        gs(corner_gx, corner_gy, 2u); // keep anti (slash)
    } else {
        gs(corner_gx, corner_gy, 0u); // tie: remove both (per paper)
    }
}
