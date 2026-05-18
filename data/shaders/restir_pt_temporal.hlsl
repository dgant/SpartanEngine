/*
Copyright(c) 2015-2026 Panos Karabelas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

//= INCLUDES ===================
#include "common.hlsl"
#include "restir_reservoir.hlsl"
//==============================

static const float TEMPORAL_MIN_CONFIDENCE = 0.1f;

float2 reproject_to_previous_frame(float2 current_uv)
{
    float2 velocity_ndc = tex_velocity.SampleLevel(GET_SAMPLER(sampler_point_clamp), current_uv, 0).xy;
    float2 velocity_uv  = velocity_ndc * float2(0.5f, -0.5f);
    return current_uv - velocity_uv;
}

// validates temporal reprojection via surface similarity + reprojection distance
bool is_temporal_sample_valid(
    float2 current_uv,
    float2 prev_uv,
    float3 current_pos,
    float3 current_normal,
    float current_depth,
    float2 screen_resolution,
    out float confidence)
{
    confidence = 0.0f;

    if (!is_valid_uv(prev_uv))
        return false;

    // reproject the current surface through the previous frame's transform and compare
    float4 prev_clip        = mul(float4(current_pos, 1.0f), get_view_projection_previous());
    float3 prev_ndc         = prev_clip.xyz / prev_clip.w;
    float2 expected_prev_uv = prev_ndc.xy * float2(0.5f, -0.5f) + 0.5f;
    float2 reproj_diff      = abs(prev_uv - expected_prev_uv) * screen_resolution;
    float  reproj_dist      = length(reproj_diff);

    float2 motion       = (current_uv - prev_uv) * screen_resolution;
    float  motion_len   = length(motion);
    float  motion_factor = saturate(motion_len / 32.0f);

    float reproj_tol = lerp(1.5f, 0.75f, motion_factor);
    if (reproj_dist > reproj_tol)
        return false;

    float normal_threshold   = lerp(0.9f, 0.97f, motion_factor);
    float3 prev_normal       = get_normal(prev_uv);
    float  normal_similarity = dot(current_normal, prev_normal);
    if (normal_similarity < normal_threshold)
        return false;

    float reproj_confidence = saturate(1.0f - reproj_dist / reproj_tol);
    float normal_confidence = saturate((normal_similarity - normal_threshold) / max(1.0f - normal_threshold, 1e-4f));
    float motion_confidence = saturate(1.0f - motion_len / 32.0f);
    confidence = reproj_confidence * normal_confidence * motion_confidence;

    return confidence >= TEMPORAL_MIN_CONFIDENCE;
}

[numthreads(THREAD_GROUP_COUNT_X, THREAD_GROUP_COUNT_Y, 1)]
void main_cs(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint2 pixel = dispatch_id.xy;
    uint resolution_x, resolution_y;
    tex_uav.GetDimensions(resolution_x, resolution_y);
    float2 resolution = float2(resolution_x, resolution_y);

    if (pixel.x >= resolution_x || pixel.y >= resolution_y)
        return;

    float2 uv = (pixel + 0.5f) / resolution;

    float depth = tex_depth.SampleLevel(GET_SAMPLER(sampler_point_clamp), uv, 0).r;
    if (depth <= 0.0f)
        return;

    float3 pos_ws    = get_position(uv);
    float3 normal_ws = get_normal(uv);
    float3 view_dir  = normalize(get_camera_position() - pos_ws);
    float4 material  = tex_material.SampleLevel(GET_SAMPLER(sampler_point_clamp), uv, 0);
    float3 albedo    = restir_srgb_to_linear_reflectance(tex_albedo.SampleLevel(GET_SAMPLER(sampler_point_clamp), uv, 0).rgb);
    float  roughness = max(material.r, 0.04f);
    float  metallic  = material.g;

    Reservoir current = unpack_reservoir(
        tex_reservoir0[pixel],
        tex_reservoir1[pixel],
        tex_reservoir2[pixel],
        tex_reservoir3[pixel],
        tex_reservoir4[pixel]
    );

    if (!is_reservoir_valid(current))
        current = create_empty_reservoir();

    uint seed = create_seed_for_pass(pixel, buffer_frame.frame, 1);

    // target_pdf of the current stream at the shading pixel (self-shift, invalid-rc allowed)
    float target_cur = target_pdf_self(current.sample, pos_ws, normal_ws, view_dir, albedo, roughness, metallic);

    // combined reservoir seeded with the current stream
    Reservoir combined   = create_empty_reservoir();
    combined.weight_sum  = 0.0f;
    combined.M           = 0.0f;
    combined.sample      = current.sample;
    combined.target_pdf  = target_cur;

    float linear_depth = linearize_depth(depth);
    float2 prev_uv     = reproject_to_previous_frame(uv);
    float  temporal_confidence = 0.0f;

    bool have_temporal = false;
    Reservoir temporal = create_empty_reservoir();
    float  target_temp          = 0.0f;
    float  jacobian_temp        = 0.0f;

    if (is_temporal_sample_valid(uv, prev_uv, pos_ws, normal_ws, linear_depth, buffer_frame.resolution_render, temporal_confidence))
    {
        float2 prev_pixel_f = prev_uv * resolution;
        bool in_bounds = prev_pixel_f.x >= 0.5f && prev_pixel_f.x < resolution.x - 0.5f &&
                         prev_pixel_f.y >= 0.5f && prev_pixel_f.y < resolution.y - 0.5f;
        if (in_bounds)
        {
            int2 prev_pixel = int2(prev_pixel_f);

            temporal = unpack_reservoir(
                tex_reservoir_prev0[prev_pixel],
                tex_reservoir_prev1[prev_pixel],
                tex_reservoir_prev2[prev_pixel],
                tex_reservoir_prev3[prev_pixel],
                tex_reservoir_prev4[prev_pixel]
            );

            if (is_reservoir_valid(temporal) && temporal.M > 0.0f && temporal.W > 0.0f)
            {
                // temporal reuse is always sub-pixel on a surface that passed the reprojection
                // gate, so approximate src_primary == pos_ws which yields jacobian ~ 1 without
                // requiring a prev-frame depth buffer
                ShiftResult shift = try_reconnection_shift(
                    temporal.sample,
                    pos_ws,
                    pos_ws,
                    normal_ws,
                    view_dir,
                    albedo,
                    roughness,
                    metallic
                );

                if (shift.ok)
                {
                    bool visible = trace_shift_visibility(temporal.sample, pos_ws, normal_ws);
                    if (visible)
                    {
                        target_temp   = max(dot(shift.f_dst, float3(0.299f, 0.587f, 0.114f)), 0.0f);
                        jacobian_temp = shift.jacobian;
                        have_temporal = (target_temp > 0.0f);
                    }
                }
            }
        }
    }

    // soft scale temporal M by reprojection confidence to fade out near surface
    // boundaries, plus a small constant decay to adapt to lighting changes; the M cap
    // bounds stale sample influence and the validity gate above already drops temporal
    // entirely for hard surface mismatches, so no time-based staleness is needed
    if (have_temporal)
    {
        float M_scale = RESTIR_TEMPORAL_DECAY * temporal_confidence;
        temporal.M    = max(temporal.M * M_scale, 0.0f);
        clamp_reservoir_M(temporal, RESTIR_M_CAP);
    }

    // generalized balance heuristic for two streams (lin 2022 eq. 25):
    //   m_i = (M_i * p_hat_dst(T_i(X_i))) / sum_j (M_j * p_hat_dst(T_j(X_j)))
    //   w_i = m_i * p_hat_dst(T_i(X_i)) * W_i_src * |J_i|
    //   W_out = sum_i w_i / p_hat_dst(Y), no extra /M_total since m_i absorbs normalization
    float denom = current.M * target_cur;
    if (have_temporal)
        denom += temporal.M * target_temp;

    float weight_cur = 0.0f;
    if (denom > 0.0f && target_cur > 0.0f)
    {
        float m_cur = (current.M * target_cur) / denom;
        weight_cur  = m_cur * target_cur * current.W;
    }

    float weight_tmp = 0.0f;
    if (have_temporal && denom > 0.0f && target_temp > 0.0f)
    {
        float m_temp = (temporal.M * target_temp) / denom;
        weight_tmp   = m_temp * target_temp * jacobian_temp * temporal.W;
    }

    combined.weight_sum = max(weight_cur, 0.0f);
    combined.M          = current.M;

    if (have_temporal)
    {
        combined.weight_sum += max(weight_tmp, 0.0f);
        combined.M          += temporal.M;

        if (combined.weight_sum > 0.0f && random_float(seed) * combined.weight_sum < weight_tmp)
        {
            combined.sample     = temporal.sample;
            combined.target_pdf = target_temp;
        }
    }

    clamp_reservoir_M(combined, RESTIR_M_CAP);

    // finalize: W = weight_sum / p_hat_dst(Y) (no /M, m_i factors already normalized)
    float final_target = target_pdf_self(combined.sample, pos_ws, normal_ws, view_dir, albedo, roughness, metallic);
    combined.target_pdf = final_target;
    combined.W          = (final_target > 0.0f) ? (combined.weight_sum / final_target) : 0.0f;

    float w_clamp = get_w_clamp_for_sample(combined.sample);
    combined.W    = min(combined.W, w_clamp);

    combined.age        = have_temporal ? (temporal.age + 1.0f) : 0.0f;
    combined.confidence = saturate(max(current.confidence, have_temporal ? temporal.confidence * temporal_confidence : 0.0f));

    float4 t0, t1, t2, t3, t4;
    pack_reservoir(combined, t0, t1, t2, t3, t4);
    tex_reservoir0[pixel] = t0;
    tex_reservoir1[pixel] = t1;
    tex_reservoir2[pixel] = t2;
    tex_reservoir3[pixel] = t3;
    tex_reservoir4[pixel] = t4;

    float3 gi = shade_reservoir_path(combined, pos_ws, normal_ws, view_dir, albedo, roughness, metallic);
    if (any(isnan(gi)) || any(isinf(gi)))
        gi = float3(0, 0, 0);

    tex_uav[pixel] = float4(gi, saturate(combined.confidence));
}
