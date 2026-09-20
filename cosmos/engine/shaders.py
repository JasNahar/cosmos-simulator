"""
GLSL shader sources.

Kept as Python strings rather than separate .glsl files so the package has
no data-file loading to get wrong. Each one is heavily commented, because
these shaders contain most of the interesting graphics ideas in the project.

Shared conventions:
  * All positions arriving here are CAMERA-RELATIVE, in metres, float32.
    The (body - camera) subtraction happened on the CPU in float64.
  * uView is a 3x3 rotation only. There is no translation left to do.
  * Depth is written logarithmically. Two lines in the vertex shader, one in
    the fragment shader, and suddenly a 1e14-metre scene has no z-fighting.
"""

# ---------------------------------------------------------------------------
# Shared snippet: logarithmic depth
# ---------------------------------------------------------------------------
# A normal depth buffer stores z'/w, which is hyperbolic in view depth: over
# 90% of the precision is spent in the closest 10% of the range. Useless when
# the range is 1 metre to 100 billion kilometres.
#
# Storing log2(depth) instead gives uniform precision *per order of magnitude*,
# which is exactly right for a scene that spans fifteen of them.
#
#   Fcoef = 2 / log2(far + 1)
#   depth = log2(1 + w) * Fcoef * 0.5          (w = view-space distance)

# ---------------------------------------------------------------------------
# BODIES -- GPU-raytraced sphere impostors
# ---------------------------------------------------------------------------
# A sphere mesh needs hundreds of triangles and still looks faceted on the
# silhouette. Instead each body is drawn as a single camera-facing QUAD --
# four vertices -- and the fragment shader solves the ray/sphere intersection
# analytically for every pixel.
#
# Result: a mathematically perfect sphere at any zoom level, with correct
# per-pixel normals and correct per-pixel depth, from 4 vertices. This is what
# makes 20,000 asteroids at 60 fps possible.

BODY_VS = """
#version 330 core

// per-vertex: the unit quad, (-1,-1) .. (1,1)
in vec2 in_corner;

// per-instance: one body
in vec3  in_center;      // camera-relative position, metres
in float in_radius;      // metres
in vec3  in_color;       // linear RGB
in float in_emissive;    // 0 = lit by stars, 1 = is a star

uniform mat3  uView;
uniform mat4  uProj;
uniform float uFcoef;        // logarithmic depth coefficient
uniform float uViewportH;    // pixels
uniform float uFocal;        // 1 / tan(fov/2)
uniform float uMinPixels;    // LOD: smallest radius we will draw, in pixels

out vec2  vLocal;        // quad coords, passed to the fragment shader
out vec3  vCenterView;   // sphere centre in view space
out float vRadius;       // true sphere radius
out float vQuadScale;    // half-size of the quad in view-space metres
out vec3  vColor;
out float vEmissive;
out float vPoint;        // 1 = too small to shade, draw as a glowing point

void main() {
    vec3 c = uView * in_center;           // view space; no translation needed
    float depth = max(-c.z, 1e-3);        // distance in front of the camera

    // --- silhouette correction -------------------------------------------
    // The visible edge of a sphere is its TANGENT circle, which is slightly
    // wider than the radius when you are close to it. Ignoring this clips
    // the edges off planets you fly up to.
    float s = in_radius;
    float d = length(c);
    if (d > in_radius * 1.0001) {
        s = in_radius / sqrt(1.0 - (in_radius * in_radius) / (d * d));
    } else {
        s = in_radius * 50.0;             // camera is inside the body
    }

    // --- level of detail --------------------------------------------------
    // How many pixels across will this be?  (H/2) * focal * r / depth.
    // Below a pixel or two there is no point solving a sphere: expand the
    // quad to a minimum size and flag it for cheap point shading instead.
    float pixels = (uViewportH * 0.5) * uFocal * s / depth;
    vPoint = 0.0;
    if (pixels < uMinPixels) {
        s = uMinPixels * depth / ((uViewportH * 0.5) * uFocal);
        vPoint = 1.0;
    }

    // The quad faces the camera by construction: in VIEW space, the screen
    // plane is simply xy, so we offset along x and y and leave z alone.
    vec3 p = c + vec3(in_corner * s, 0.0);

    gl_Position = uProj * vec4(p, 1.0);

    // --- logarithmic depth (vertex half) ---------------------------------
    float w = max(gl_Position.w, 1e-6);
    gl_Position.z = (log2(1.0 + w) * uFcoef - 1.0) * w;

    vLocal      = in_corner;
    vCenterView = c;
    vRadius     = in_radius;
    vQuadScale  = s;
    vColor      = in_color;
    vEmissive   = in_emissive;
}
"""

BODY_FS = """
#version 330 core

in vec2  vLocal;
in vec3  vCenterView;
in float vRadius;
in float vQuadScale;
in vec3  vColor;
in float vEmissive;
in float vPoint;

#define MAX_LIGHTS 8
uniform int   uLightCount;
uniform vec3  uLightPos[MAX_LIGHTS];    // view space, metres
uniform vec3  uLightColor[MAX_LIGHTS];
uniform float uLightLum[MAX_LIGHTS];    // normalised so 1.0 = Sun at 1 AU
uniform float uFcoef;
uniform float uExposure;
uniform float uAmbient;

out vec4 fragColor;

void main() {
    float r2 = dot(vLocal, vLocal);
    if (r2 > 1.0) discard;                       // outside the inscribed disc

    // ---- cheap path: object is a couple of pixels across ----------------
    if (vPoint > 0.5) {
        // A soft gaussian-ish dot. No normal, no lighting -- at this size
        // the difference is invisible and the cost is not.
        float a = exp(-4.0 * r2);
        vec3 col = vColor * (vEmissive > 0.5 ? 2.0 : 0.85);
        fragColor = vec4(col * a * uExposure, a);
        float w = max(-vCenterView.z, 1e-6);
        gl_FragDepth = log2(1.0 + w) * uFcoef * 0.5;
        return;
    }

    // ---- ray/sphere intersection, analytically --------------------------
    // The quad may be larger than the sphere (silhouette correction), so
    // re-derive where on the sphere this pixel actually lands.
    vec2 xy = vLocal * vQuadScale;               // view-space offset, metres
    float rr = dot(xy, xy) / (vRadius * vRadius);
    if (rr > 1.0) discard;                       // this pixel misses the sphere

    // Surface normal: the front hemisphere of a unit sphere.
    vec3 n = vec3(xy / vRadius, sqrt(1.0 - rr));
    vec3 p = vCenterView + n * vRadius;          // the surface point itself

    vec3 col;
    if (vEmissive > 0.5) {
        // A star. Limb darkening: the edge of a stellar disc is dimmer than
        // the centre because you are looking through more atmosphere at a
        // shallow angle. The Sun really does look like this.
        float mu = n.z;
        col = vColor * (0.45 + 0.55 * pow(max(mu, 0.0), 0.45)) * 3.0;
    } else {
        col = vec3(0.0);
        for (int i = 0; i < uLightCount; ++i) {
            vec3 Lv = uLightPos[i] - p;
            float dist2 = max(dot(Lv, Lv), 1.0);
            vec3 L = Lv * inversesqrt(dist2);

            // Lambert's cosine law. A beam striking at an angle spreads over
            // a larger area, so each patch receives less energy -- and the
            // spreading factor is exactly cos(theta) = dot(n, L).
            float ndl = max(dot(n, L), 0.0);

            // Inverse-square falloff, normalised so that a Sun-luminosity
            // star at 1 AU gives irradiance 1.0.
            float irradiance = uLightLum[i] / dist2;

            col += vColor * uLightColor[i] * ndl * irradiance;
        }
        col += vColor * uAmbient;    // faint fill so night sides aren't voids
    }

    // ---- logarithmic depth (fragment half) ------------------------------
    // Uses the depth of the SURFACE point, not the quad, so spheres
    // intersect and occlude each other correctly.
    float w = max(-p.z, 1e-6);
    gl_FragDepth = log2(1.0 + w) * uFcoef * 0.5;

    fragColor = vec4(col * uExposure, 1.0);
}
"""

# ---------------------------------------------------------------------------
# STARFIELD -- the background sky
# ---------------------------------------------------------------------------
# Drawn as GL points at a huge fixed distance, so they never move relative to
# the camera's position (only its rotation), exactly like real stars.

STARS_VS = """
#version 330 core
in vec3  in_dir;         // unit vector: direction on the celestial sphere
in vec3  in_color;       // blackbody colour for the star's temperature
in float in_size;        // apparent brightness -> point size

uniform mat3  uView;
uniform mat4  uProj;
uniform float uFcoef;
uniform float uDistance;

out vec3  vColor;
out float vBright;

void main() {
    vec3 p = uView * (in_dir * uDistance);
    gl_Position = uProj * vec4(p, 1.0);
    float w = max(gl_Position.w, 1e-6);
    gl_Position.z = (log2(1.0 + w) * uFcoef - 1.0) * w;
    gl_PointSize = in_size;
    vColor  = in_color;
    vBright = clamp(in_size / 3.0, 0.25, 1.6);
}
"""

STARS_FS = """
#version 330 core
in vec3  vColor;
in float vBright;
out vec4 fragColor;

void main() {
    // gl_PointCoord runs 0..1 across the point sprite.
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float d2 = dot(q, q);
    if (d2 > 1.0) discard;
    // Airy-disc-ish falloff: a bright core with a soft halo.
    float a = exp(-5.0 * d2);
    fragColor = vec4(vColor * vBright * a, a);
}
"""

# ---------------------------------------------------------------------------
# TRAILS -- orbit history
# ---------------------------------------------------------------------------

TRAIL_VS = """
#version 330 core
in vec3  in_pos;         // camera-relative, metres
in float in_age;         // 0 = newest, 1 = oldest

uniform mat3  uView;
uniform mat4  uProj;
uniform float uFcoef;
uniform vec3  uColor;

out vec4 vColor;

void main() {
    vec3 p = uView * in_pos;
    gl_Position = uProj * vec4(p, 1.0);
    float w = max(gl_Position.w, 1e-6);
    gl_Position.z = (log2(1.0 + w) * uFcoef - 1.0) * w;
    vColor = vec4(uColor, (1.0 - in_age) * 0.55);
}
"""

TRAIL_FS = """
#version 330 core
in vec4 vColor;
out vec4 fragColor;
void main() { fragColor = vColor; }
"""

# ---------------------------------------------------------------------------
# GRAVITATIONAL LENSING -- the black hole post-process
# ---------------------------------------------------------------------------
# The scene is rendered into a texture, then this fullscreen pass bends the
# light around each black hole before it reaches the "eye".
#
# The physics: a light ray passing a mass M at impact parameter b is deflected
# by Einstein's angle
#
#       alpha = 4 G M / (c^2 b) = 2 r_s / b
#
# (Newton, treating light as a particle, predicts exactly half this. The
# factor of two was what the 1919 Eddington eclipse expedition measured, and
# it made Einstein famous.)
#
# So for a pixel at angular offset theta from the hole, the light arriving
# there actually set off from angular offset theta - alpha(theta). We look up
# THAT direction in the rendered scene. Rays with b below the critical value
#
#       b_crit = (3 sqrt(3) / 2) r_s ~ 2.598 r_s
#
# are captured and never escape -- that is the black hole's SHADOW, and it is
# noticeably larger than the event horizon itself. The bright rim just outside
# it is the PHOTON RING: light that orbited the hole one or more times before
# escaping toward the camera.
#
# This is a screen-space approximation of a genuinely 4D problem. A full
# treatment integrates null geodesics of the Schwarzschild metric per pixel.
# The weak-field deflection used here is accurate to a few percent outside
# ~5 r_s and captures the shadow, the ring, and the Einstein ring correctly.

LENS_VS = """
#version 330 core
in vec2 in_pos;
out vec2 vUV;
void main() {
    vUV = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

LENS_FS = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uScene;

#define MAX_BH 4
uniform int   uBhCount;
uniform vec2  uBhNdc[MAX_BH];      // screen position, normalised device coords
uniform float uBhRs[MAX_BH];       // Schwarzschild radius, metres
uniform float uBhDist[MAX_BH];     // distance from camera, metres
uniform vec3  uBhDiskColor[MAX_BH];

uniform float uTanHalfFov;
uniform float uAspect;
uniform float uBehind[MAX_BH];     // 1.0 if the hole is behind the camera

// NDC <-> angular offset (small-angle; exact to the same order as the
// deflection formula itself).
vec2 ndc_to_angle(vec2 n) {
    return vec2(n.x * uAspect * uTanHalfFov, n.y * uTanHalfFov);
}
vec2 angle_to_ndc(vec2 a) {
    return vec2(a.x / (uAspect * uTanHalfFov), a.y / uTanHalfFov);
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec2 ang = ndc_to_angle(ndc);

    vec3 extra = vec3(0.0);
    bool shadowed = false;

    for (int i = 0; i < uBhCount; ++i) {
        if (uBehind[i] > 0.5) continue;

        vec2 delta = ang - ndc_to_angle(uBhNdc[i]);
        float theta = length(delta);
        if (theta < 1e-9) { shadowed = true; continue; }

        float rs = uBhRs[i];
        float b  = uBhDist[i] * theta;          // impact parameter, metres
        float b_crit = 2.598076 * rs;           // 3*sqrt(3)/2 * rs

        if (b < b_crit) {                       // captured: the shadow
            shadowed = true;
            continue;
        }

        // Einstein deflection, then trace back to the source direction.
        //
        // beta = theta - alpha(theta) is the LENS EQUATION. Outside the
        // Einstein radius theta_E = sqrt(2 rs / d) it gives the ordinary
        // ("primary") image. Inside, beta goes negative -- and that is not
        // an error: it is the SECONDARY image, the light that passed the
        // far side of the hole and was bent back toward us, appearing
        // inverted on the opposite side. Letting the sign flow through the
        // direction vector reproduces it for free.
        float alpha = 2.0 * rs / b;
        float theta_src = theta - alpha;

        ang = ndc_to_angle(uBhNdc[i]) + (delta / theta) * theta_src;

        // Photon ring: light that looped the hole before escaping. Brightest
        // just outside b_crit and falling off fast.
        float x = b / b_crit;
        float ring = exp(-28.0 * (x - 1.02) * (x - 1.02));
        extra += uBhDiskColor[i] * ring * 1.6;
    }

    if (shadowed) {
        fragColor = vec4(extra, 1.0);           // pure black plus any ring
        return;
    }

    vec2 uv = angle_to_ndc(ang) * 0.5 + 0.5;
    vec3 col;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        col = vec3(0.0);                        // deflected in from off-screen
    } else {
        col = texture(uScene, uv).rgb;
    }
    fragColor = vec4(col + extra, 1.0);
}
"""

# ---------------------------------------------------------------------------
# TONEMAP / HUD BLIT
# ---------------------------------------------------------------------------

BLIT_VS = LENS_VS

TONEMAP_FS = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uScene;

void main() {
    vec3 c = texture(uScene, vUV).rgb;
    // Reinhard-style tonemap. Real astronomical brightness spans a huge
    // dynamic range (the Sun is ~1e10 times brighter than a dim asteroid);
    // this compresses it into something a monitor can show without the
    // bright things simply clipping to flat white discs.
    c = c / (c + vec3(1.0));
    c = pow(c, vec3(1.0 / 2.2));      // linear -> sRGB
    fragColor = vec4(c, 1.0);
}
"""

HUD_FS = """
#version 330 core
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uHud;
void main() {
    vec4 c = texture(uHud, vec2(vUV.x, 1.0 - vUV.y));
    if (c.a < 0.01) discard;
    fragColor = c;
}
"""
