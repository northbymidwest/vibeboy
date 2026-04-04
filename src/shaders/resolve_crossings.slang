// Resolve diagonal crossings in the similarity graph.
// Each thread processes one 2x2 pixel block.
// Three heuristics vote to decide which diagonal to keep:
//   1. Curve length -- longer connected chain wins
//   2. Sparse pixels -- smaller component (BFS in 8x8 window) wins
//   3. Islands -- valence-1 endpoint gets fixed weight 5 (early return on first found)
// Fully-connected blocks (all 4 cardinal edges) remove both diagonals.
// Single diagonals in fully-connected blocks are also removed.
// Ties keep both diagonals (preserves thin features like Space Invaders sprites).

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
    uint c = 0u;
    uint m = mask;
    while (m != 0u) { c += m & 1u; m >>= 1u; }
    return c;
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

// Simultaneous A/B component labeling in 8x8 window around the crossing.
void sparse_pixel_sizes(uint bx, uint by, out uint sizeA, out uint sizeB) {
    int labels[64];
    for (int i = 0; i < 64; i++) labels[i] = 0;
    labels[3*8+3] = 1;
    labels[3*8+4] = 2;
    labels[4*8+3] = 2;
    labels[4*8+4] = 1;

    sizeA = 0u;
    sizeB = 0u;

    uint q_col[64], q_row[64];
    uint head = 0u, tail = 0u;
    q_col[tail] = 3u; q_row[tail] = 3u; tail++;
    q_col[tail] = 4u; q_row[tail] = 3u; tail++;
    q_col[tail] = 4u; q_row[tail] = 4u; tail++;
    q_col[tail] = 3u; q_row[tail] = 4u; tail++;

    static const int dcol[8] = { -1, -1, -1, 0, 1, 1, 1, 0 };
    static const int drow[8] = { -1, 0, 1, 1, 1, 0, -1, -1 };
    static const uint dirs[8] = { DIR_NW, DIR_W, DIR_SW, DIR_S, DIR_SE, DIR_E, DIR_NE, DIR_N };

    while (head < tail) {
        uint col = q_col[head];
        uint row = q_row[head];
        head++;
        int label = labels[row*8u+col];

        int px = int(bx) + int(col) - 3;
        int py = int(by) + int(row) - 3;
        if (px < 0 || py < 0 || px >= int(u.img_w) || py >= int(u.img_h)) continue;

        uint vmask = valence_mask(uint(px), uint(py));

        for (uint d = 0u; d < 8u; d++) {
            if ((vmask & dirs[d]) == 0u) continue;
            int nc = int(col) + dcol[d];
            int nr = int(row) + drow[d];
            if (nc < 0 || nc >= 8 || nr < 0 || nr >= 8) continue;
            if (labels[nr*8+nc] != 0) continue;

            int npx = int(bx) + nc - 3;
            int npy = int(by) + nr - 3;
            if (npx < 0 || npy < 0 || npx >= int(u.img_w) || npy >= int(u.img_h)) continue;

            labels[nr*8+nc] = label;
            if (label == 1) sizeA++;
            else sizeB++;

            if (tail < 64u) {
                q_col[tail] = uint(nc);
                q_row[tail] = uint(nr);
                tail++;
            }
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
        gs(corner_gx, corner_gy, 3u); // tie: keep both (preserves thin features)
    }
}
