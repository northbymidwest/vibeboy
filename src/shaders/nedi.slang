// NEDI -- New Edge-Directed Interpolation (2x upscaling).
// Local covariance estimation for adaptive interpolation weights.

import modules.color;
import modules.pixel_source;

struct Uniforms {
    uint src_w;
    uint src_h;
    uint out_w;
    uint out_h;
    uint iscale;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::binding(0, 2)]] ConstantBuffer<Uniforms> u;

[[vk::binding(0, 0)]] StructuredBuffer<uint> pixels;
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]] WTexture2D<float4> output_img;

float4 fetch(int2 coord) { return px_rgba(pixels, int(u.src_w), int(u.src_h), coord); }

float luma(int2 coord) {
    coord = clamp(coord, int2(0), int2(int(u.src_w) - 1, int(u.src_h) - 1));
    uint c = pixels[coord.y * int(u.src_w) + coord.x];
    return 0.299 * float((c >> 16u) & 0xFFu)
         + 0.587 * float((c >> 8u)  & 0xFFu)
         + 0.114 * float( c         & 0xFFu);
}

// Solve 4x4 linear system via Gaussian elimination with partial pivoting.
// Returns false if singular. Result in x[0..3].
bool solve4x4(inout float m[4][5], out float x[4]) {
    x[0] = 0.0; x[1] = 0.0; x[2] = 0.0; x[3] = 0.0;
    for (int col = 0; col < 4; col++) {
        // Partial pivot
        int max_row = col;
        float max_val = abs(m[col][col]);
        for (int row = col + 1; row < 4; row++) {
            if (abs(m[row][col]) > max_val) {
                max_val = abs(m[row][col]);
                max_row = row;
            }
        }
        if (max_val < 1e-10) return false;
        if (max_row != col) {
            for (int j = 0; j < 5; j++) {
                float tmp = m[col][j];
                m[col][j] = m[max_row][j];
                m[max_row][j] = tmp;
            }
        }
        float pivot = m[col][col];
        for (int row = col + 1; row < 4; row++) {
            float factor = m[row][col] / pivot;
            for (int j = col; j < 5; j++) {
                m[row][j] -= factor * m[col][j];
            }
        }
    }
    // Back substitution
    for (int i = 3; i >= 0; i--) {
        float s = m[i][4];
        for (int j = i + 1; j < 4; j++) {
            s -= m[i][j] * x[j];
        }
        if (abs(m[i][i]) < 1e-10) return false;
        x[i] = s / m[i][i];
    }
    return true;
}

// Compute NEDI diamond weights for interpolation point at (x+0.5, y+0.5).
// d0=(x,y) d1=(x+1,y) d2=(x,y+1) d3=(x+1,y+1)
float4 nedi_diamond_weights(int2 p) {
    // Training samples from nearby checkerboard-pattern pixels
    float c_arr[12][4];
    float y_arr[12];
    int count = 0;

    for (int dy = -1; dy <= 2; dy++) {
        for (int dx = -1; dx <= 2; dx++) {
            if (((dx + dy) & 1) != 0) continue;
            if (count >= 12) break;
            int2 cp = p + int2(dx, dy);
            float center = luma(cp);
            float n0 = luma(cp + int2(-1, -1));
            float n1 = luma(cp + int2( 1, -1));
            float n2 = luma(cp + int2(-1,  1));
            float n3 = luma(cp + int2( 1,  1));
            c_arr[count][0] = n0;
            c_arr[count][1] = n1;
            c_arr[count][2] = n2;
            c_arr[count][3] = n3;
            y_arr[count] = center;
            count++;
        }
    }

    if (count < 4) return float4(0.25);

    // Build C^T * C (4x4) and C^T * y (4x1)
    float ctc[4][4];
    float cty[4];
    for (int i = 0; i < 4; i++) {
        cty[i] = 0.0;
        for (int j = 0; j < 4; j++) ctc[i][j] = 0.0;
    }
    for (int k = 0; k < count; k++) {
        for (int i = 0; i < 4; i++) {
            cty[i] += c_arr[k][i] * y_arr[k];
            for (int j = 0; j < 4; j++) {
                ctc[i][j] += c_arr[k][i] * c_arr[k][j];
            }
        }
    }
    // Regularization
    for (int i = 0; i < 4; i++) ctc[i][i] += 0.001;

    // Build augmented matrix
    float m[4][5];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) m[i][j] = ctc[i][j];
        m[i][4] = cty[i];
    }

    float w[4];
    if (!solve4x4(m, w)) return float4(0.25);

    // Normalize
    float s = w[0] + w[1] + w[2] + w[3];
    if (abs(s) < 0.01) return float4(0.25);
    float inv = 1.0 / s;

    // Clamp to avoid ringing, then re-normalize
    w[0] = clamp(w[0] * inv, -0.25, 1.25);
    w[1] = clamp(w[1] * inv, -0.25, 1.25);
    w[2] = clamp(w[2] * inv, -0.25, 1.25);
    w[3] = clamp(w[3] * inv, -0.25, 1.25);
    s = w[0] + w[1] + w[2] + w[3];
    if (abs(s) < 0.01) return float4(0.25);
    inv = 1.0 / s;
    return float4(w[0] * inv, w[1] * inv, w[2] * inv, w[3] * inv);
}

float4 blend4(float4 c0, float4 c1, float4 c2, float4 c3, float4 w) {
    return c0 * w.x + c1 * w.y + c2 * w.z + c3 * w.w;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    int2 src_i = int2(opx / 2u, opy / 2u);
    int2 sub = int2(opx & 1u, opy & 1u);

    float4 result;
    if (sub.x == 0 && sub.y == 0) {
        // Top-left: original pixel
        result = fetch(src_i);
    } else if (sub.x == 1 && sub.y == 0) {
        // Top-right: horizontal bilinear
        result = lerp(fetch(src_i), fetch(src_i + int2(1, 0)), 0.5);
    } else if (sub.x == 0 && sub.y == 1) {
        // Bottom-left: vertical bilinear
        result = lerp(fetch(src_i), fetch(src_i + int2(0, 1)), 0.5);
    } else {
        // Bottom-right: NEDI diamond interpolation
        float4 d0 = fetch(src_i);
        float4 d1 = fetch(src_i + int2(1, 0));
        float4 d2 = fetch(src_i + int2(0, 1));
        float4 d3 = fetch(src_i + int2(1, 1));
        float4 w = nedi_diamond_weights(src_i);
        result = blend4(d0, d1, d2, d3, w);
    }

    output_img.Store(int2(opx, opy), result);
}
