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

//= includes =========
#include "common.hlsl"
//====================


/*------------------------------------------------------------------------------
    constants
------------------------------------------------------------------------------*/
// camera/lens parameters
static const float SENSOR_HEIGHT_MM = 24.0f;  // full-frame 35mm sensor
static const float FOCAL_LENGTH_MM  = 50.0f;  // standard lens

// blur parameters
static const float COC_CLAMP_FACTOR   = 0.8f;   // prevent excessive blur
static const float IN_FOCUS_THRESHOLD = 0.5f;   // coc below this = sharp
static const int   SAMPLE_COUNT       = 48;     // bokeh quality (higher = smoother)
static const float GOLDEN_ANGLE       = 2.39996323f;

// auto-focus parameters
static const int   FOCUS_SAMPLES      = 16;     // depth samples for focus calculation

// derived constants (compile-time)
static const float INV_SAMPLE_COUNT  = 1.0f / (float)SAMPLE_COUNT;

float get_max_coc_radius()       { return max(pass_get_f3_value2().x, 0.1f); }
float get_near_scale()           { return max(pass_get_f3_value2().y, 0.0f); }
float get_far_scale()            { return max(pass_get_f3_value2().z, 0.0f); }
float get_focus_region()         { return max(pass_get_f4_value().x, 0.001f); }
float get_center_weight_bias()   { return max(pass_get_f4_value().y, 0.0f); }
float get_outlier_threshold()    { return max(pass_get_f4_value().z, 0.001f); }
float get_leak_prevention()      { return saturate(pass_get_f4_value().w); }

/*------------------------------------------------------------------------------
    lens constants computed once per group then read by every thread
------------------------------------------------------------------------------*/
struct lens_t
{
    float focus_distance;  // s, focus distance in meters clamped above focal length
    float coc_factor;      // aperture_diameter * f * resolution.y / sensor_m / abs(s - f)
};

groupshared lens_t gs_lens;

/*------------------------------------------------------------------------------
    fast signed coc using precomputed lens factor
    sign is negative for foreground (d < s) and positive for background (d > s)
------------------------------------------------------------------------------*/
float compute_coc_signed(float depth, lens_t lens)
{
    float d         = max(depth, 0.01f);
    float s_minus_d = lens.focus_distance - d;
    float coc_pix   = abs(s_minus_d) * lens.coc_factor / d;
    bool  is_near   = s_minus_d > 0.0f;
    float scale     = is_near ? get_near_scale() : get_far_scale();
    coc_pix         = min(coc_pix * scale, get_max_coc_radius() * COC_CLAMP_FACTOR);
    return is_near ? -coc_pix : coc_pix;
}

/*------------------------------------------------------------------------------
    robust auto-focus with weighted sampling
    
    uses a spiral pattern centered on screen, weights center samples higher,
    and rejects outliers to avoid focusing on background through holes
------------------------------------------------------------------------------*/
float compute_focus_distance(float2 resolution)
{
    float2 center = float2(0.5f, 0.5f);
    
    float depths[FOCUS_SAMPLES];
    float weights[FOCUS_SAMPLES];
    float weight_sum = 0.0f;
    
    [unroll]
    for (int i = 0; i < FOCUS_SAMPLES; i++)
    {
        float t      = (float)i / (float)(FOCUS_SAMPLES - 1);
        float angle  = i * GOLDEN_ANGLE;
        float focus_region = get_focus_region();
        float radius = sqrt(t) * focus_region;
        
        float sin_a, cos_a;
        sincos(angle, sin_a, cos_a);
        float2 offset = float2(cos_a, sin_a) * radius;
        float2 uv     = center + offset;
        
        depths[i]    = get_linear_depth(uv * buffer_frame.resolution_scale);
        float dist_n = length(offset) / focus_region;
        weights[i]   = exp(-dist_n * dist_n * get_center_weight_bias());
        weight_sum  += weights[i];
    }
    
    float weighted_avg = 0.0f;
    [unroll]
    for (int j = 0; j < FOCUS_SAMPLES; j++)
    {
        weighted_avg += depths[j] * weights[j];
    }
    weighted_avg /= max(weight_sum, FLT_MIN);
    
    float refined_sum    = 0.0f;
    float refined_weight = 0.0f;
    float inv_avg        = 1.0f / max(weighted_avg, 0.1f);
    
    [unroll]
    for (int k = 0; k < FOCUS_SAMPLES; k++)
    {
        float deviation = abs(depths[k] - weighted_avg) * inv_avg;
        float outlier_threshold = get_outlier_threshold();
        if (deviation < outlier_threshold)
        {
            float confidence  = 1.0f - deviation / outlier_threshold;
            confidence       *= confidence;
            float w           = weights[k] * confidence;
            refined_sum      += depths[k] * w;
            refined_weight   += w;
        }
    }
    
    return (refined_weight > FLT_MIN) ? (refined_sum / refined_weight) : weighted_avg;
}

/*------------------------------------------------------------------------------
    main bokeh blur with gather sampling
------------------------------------------------------------------------------*/
float3 bokeh_gather(float2 uv, float center_coc, float center_depth, lens_t lens, float2 texel_size, float2 resolution)
{
    float blur_radius = abs(center_coc);

    float3 color_sum  = tex.SampleLevel(samplers[sampler_bilinear_clamp], uv, 0).rgb;
    float  weight_sum = 1.0f;
    
    bool center_is_fg = center_coc < 0.0f;

    // randomize starting angle per pixel for temporal stability with taa
    float angle = noise_interleaved_gradient(uv * resolution, true) * PI2;
    
    [loop]
    for (int i = 0; i < SAMPLE_COUNT; i++)
    {
        angle += GOLDEN_ANGLE;
        
        float t = (float)(i + 1) * INV_SAMPLE_COUNT;
        float r = sqrt(t) * blur_radius;
        
        float sin_a, cos_a;
        sincos(angle, sin_a, cos_a);
        float2 offset    = float2(cos_a, sin_a) * r * texel_size;
        float2 sample_uv = uv + offset;
        
        if (!is_valid_uv(sample_uv))
            continue;
        
        float3 sample_color = tex.SampleLevel(samplers[sampler_bilinear_clamp], sample_uv, 0).rgb;
        float  sample_depth = get_linear_depth(sample_uv * buffer_frame.resolution_scale);
        float  sample_coc   = compute_coc_signed(sample_depth, lens);
        float  abs_sample_coc = abs(sample_coc);

        // inlined sample_weight
        float effective_coc = max(abs_sample_coc, blur_radius);
        float coverage      = saturate(1.0f - r / max(effective_coc, FLT_MIN));
        float depth_weight  = (sample_depth > center_depth && center_is_fg) ? get_leak_prevention() : 1.0f;
        float scatter       = lerp(0.3f, 1.0f, saturate(abs_sample_coc / max(get_max_coc_radius() * 0.3f, 0.001f)));
        float w             = coverage * depth_weight * scatter;
        
        color_sum  += sample_color * w;
        weight_sum += w;
    }
    
    return color_sum / max(weight_sum, FLT_MIN);
}

/*------------------------------------------------------------------------------
    compute shader entry point
------------------------------------------------------------------------------*/
[numthreads(THREAD_GROUP_COUNT_X, THREAD_GROUP_COUNT_Y, 1)]
void main_cs(uint3 thread_id : SV_DispatchThreadID, uint group_index : SV_GroupIndex)
{
    float2 resolution;
    tex_uav.GetDimensions(resolution.x, resolution.y);

    // one thread per group computes the lens constants and shares them with the rest
    if (group_index == 0)
    {
        float aperture_fstop    = pass_get_f3_value().x;
        float strength          = max(pass_get_f3_value().y, 0.0f);
        float f                 = FOCAL_LENGTH_MM * 0.001f;
        float aperture_diameter = f / max(aperture_fstop, 1.0f);
        float sensor_m          = SENSOR_HEIGHT_MM * 0.001f;
        float pixels_per_meter  = resolution.y / sensor_m;
        float manual_focus      = pass_get_f3_value().z;
        float focus_distance    = manual_focus > 0.0f ? manual_focus : compute_focus_distance(resolution);
        float s                 = max(focus_distance, f + 0.01f);

        gs_lens.focus_distance = s;
        gs_lens.coc_factor     = ((aperture_diameter * f * pixels_per_meter) / (abs(s - f) + FLT_MIN)) * strength;
    }
    GroupMemoryBarrierWithGroupSync();

    if (any(thread_id.xy >= uint2(resolution)))
        return;

    lens_t lens       = gs_lens;
    float2 uv         = (thread_id.xy + 0.5f) / resolution;
    float  depth      = get_linear_depth(uv * buffer_frame.resolution_scale);
    float  coc        = compute_coc_signed(depth, lens);
    float  blur_radius = abs(coc);

    // in-focus passthrough, smoothstep blend below 0.5 px is sub-perceptual so we skip the gather entirely
    if (blur_radius < IN_FOCUS_THRESHOLD)
    {
        tex_uav[thread_id.xy] = tex[thread_id.xy];
        return;
    }

    float2 texel_size = 1.0f / resolution;
    float3 blurred    = bokeh_gather(uv, coc, depth, lens, texel_size, resolution);

    float4 original = tex[thread_id.xy];
    float  blend    = smoothstep(0.0f, 1.0f, blur_radius / get_max_coc_radius());
    float3 result   = lerp(original.rgb, blurred, blend);

    tex_uav[thread_id.xy] = float4(result, original.a);
}
