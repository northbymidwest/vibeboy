// MMPX -- Style-Preserving Pixel-Art Magnification (2x).
// Algorithm by Morgan McGuire & Mara Gagiu (2021).

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
[[vk::binding(0, 2)]]
#ifdef DXIL_TARGET
ConstantBuffer<Uniforms> u : register(b0, space2);
#else
ConstantBuffer<Uniforms> u;
#endif

[[vk::binding(0, 0)]]
#ifdef DXIL_TARGET
StructuredBuffer<uint> pixels : register(t0, space0);
#else
StructuredBuffer<uint> pixels;
#endif
[[vk::image_format("rgba8")]]
[[vk::binding(0, 1)]]
#ifdef DXIL_TARGET
WTexture2D<float4> output_img : register(u0, space1);
#else
WTexture2D<float4> output_img;
#endif

uint fetch_raw(int2 coord) { return px_raw(pixels, int(u.src_w), int(u.src_h), coord); }

// Luma with alpha weighting (transparent = bright background)
uint luma(uint c) {
    uint r = (c >> 16u) & 0xFFu;
    uint g = (c >> 8u)  & 0xFFu;
    uint b =  c         & 0xFFu;
    uint a = (c >> 24u) & 0xFFu;
    return (r + g + b + 1u) * (256u - a);
}

bool all_eq2(uint v, uint a0, uint a1) {
    return ((v ^ a0) | (v ^ a1)) == 0u;
}

bool all_eq3(uint v, uint a0, uint a1, uint a2) {
    return ((v ^ a0) | (v ^ a1) | (v ^ a2)) == 0u;
}

bool all_eq4(uint v, uint a0, uint a1, uint a2, uint a3) {
    return ((v ^ a0) | (v ^ a1) | (v ^ a2) | (v ^ a3)) == 0u;
}

bool any_eq3(uint v, uint a0, uint a1, uint a2) {
    return v == a0 || v == a1 || v == a2;
}

bool none_eq2(uint v, uint a0, uint a1) {
    return v != a0 && v != a1;
}

bool none_eq4(uint v, uint a0, uint a1, uint a2, uint a3) {
    return v != a0 && v != a1 && v != a2 && v != a3;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID) {
    uint opx = tid.x;
    uint opy = tid.y;
    if (opx >= u.out_w || opy >= u.out_h) return;

    // Map output pixel to source pixel + sub-pixel within 2x2 block
    int2 src_i = int2(opx / 2u, opy / 2u);
    int2 sub = int2(opx & 1u, opy & 1u);
    int ix = src_i.x;
    int iy = src_i.y;

    // 3x3 neighborhood
    uint A = fetch_raw(int2(ix - 1, iy - 1));
    uint B = fetch_raw(int2(ix,     iy - 1));
    uint C = fetch_raw(int2(ix + 1, iy - 1));
    uint D = fetch_raw(int2(ix - 1, iy));
    uint E = fetch_raw(int2(ix,     iy));
    uint F = fetch_raw(int2(ix + 1, iy));
    uint G = fetch_raw(int2(ix - 1, iy + 1));
    uint H = fetch_raw(int2(ix,     iy + 1));
    uint I = fetch_raw(int2(ix + 1, iy + 1));

    // Default: nearest neighbor
    uint J = E, K = E, L = E, M = E;

    // Early exit if neighborhood is uniform
    if (((A ^ E) | (B ^ E) | (C ^ E) | (D ^ E) |
         (F ^ E) | (G ^ E) | (H ^ E) | (I ^ E)) != 0u)
    {
        // Diamond points
        uint P = fetch_raw(int2(ix,     iy - 2));
        uint Q = fetch_raw(int2(ix - 2, iy));
        uint R = fetch_raw(int2(ix + 2, iy));
        uint S = fetch_raw(int2(ix,     iy + 2));

        uint Bl = luma(B), Dl = luma(D), El = luma(E), Fl = luma(F), Hl = luma(H);

        // 1:1 slope rules
        if (D == B && D != H && D != F
            && (El >= Dl || E == A)
            && any_eq3(E, A, C, G)
            && (El < Dl || A != D || E != P || E != Q))
            J = D;

        if (B == F && B != D && B != H
            && (El >= Bl || E == C)
            && any_eq3(E, A, C, I)
            && (El < Bl || C != B || E != P || E != R))
            K = B;

        if (H == D && H != F && H != B
            && (El >= Hl || E == G)
            && any_eq3(E, A, G, I)
            && (El < Hl || G != H || E != S || E != Q))
            L = H;

        if (F == H && F != B && F != D
            && (El >= Fl || E == I)
            && any_eq3(E, C, G, I)
            && (El < Fl || I != H || E != R || E != S))
            M = F;

        // Intersection rules (structural)
        if (E != F && all_eq4(E, C, I, D, Q) && all_eq2(F, B, H)
            && F != fetch_raw(int2(ix + 3, iy)))
        { K = F; M = F; }

        if (E != D && all_eq4(E, A, G, F, R) && all_eq2(D, B, H)
            && D != fetch_raw(int2(ix - 3, iy)))
        { J = D; L = D; }

        if (E != H && all_eq4(E, G, I, B, P) && all_eq2(H, D, F)
            && H != fetch_raw(int2(ix, iy + 3)))
        { L = H; M = H; }

        if (E != B && all_eq4(E, A, C, H, S) && all_eq2(B, D, F)
            && B != fetch_raw(int2(ix, iy - 3)))
        { J = B; K = B; }

        // Intersection rules (triangle tip)
        if (Bl < El && all_eq4(E, G, H, I, S) && none_eq4(E, A, D, C, F))
        { J = B; K = B; }

        if (Hl < El && all_eq4(E, A, B, C, P) && none_eq4(E, D, G, I, F))
        { L = H; M = H; }

        if (Fl < El && all_eq4(E, A, D, G, Q) && none_eq4(E, B, C, I, H))
        { K = F; M = F; }

        if (Dl < El && all_eq4(E, C, F, I, R) && none_eq4(E, B, A, G, H))
        { J = D; L = D; }

        // 2:1 slope rules
        if (H != B) {
            if (H != A && H != E && H != C) {
                if (all_eq3(H, G, F, R)
                    && none_eq2(H, D, fetch_raw(int2(ix + 2, iy - 1))))
                    L = M;
                if (all_eq3(H, I, D, Q)
                    && none_eq2(H, F, fetch_raw(int2(ix - 2, iy - 1))))
                    M = L;
            }
            if (B != I && B != G && B != E) {
                if (all_eq3(B, A, F, R)
                    && none_eq2(B, D, fetch_raw(int2(ix + 2, iy + 1))))
                    J = K;
                if (all_eq3(B, C, D, Q)
                    && none_eq2(B, F, fetch_raw(int2(ix - 2, iy + 1))))
                    K = J;
            }
        }

        if (F != D) {
            if (D != I && D != E && D != C) {
                if (all_eq3(D, A, H, S)
                    && none_eq2(D, B, fetch_raw(int2(ix + 1, iy + 2))))
                    J = L;
                if (all_eq3(D, G, B, P)
                    && none_eq2(D, H, fetch_raw(int2(ix + 1, iy - 2))))
                    L = J;
            }
            if (F != E && F != A && F != G) {
                if (all_eq3(F, C, H, S)
                    && none_eq2(F, B, fetch_raw(int2(ix - 1, iy + 2))))
                    K = M;
                if (all_eq3(F, I, B, P)
                    && none_eq2(F, H, fetch_raw(int2(ix - 1, iy - 2))))
                    M = K;
            }
        }
    }

    // Select output pixel from the 2x2 block
    uint result;
    if (sub.x == 0 && sub.y == 0) result = J;
    else if (sub.x == 1 && sub.y == 0) result = K;
    else if (sub.x == 0 && sub.y == 1) result = L;
    else result = M;

    output_img.Store(int2(opx, opy), (result).unpack_rgba());
}
