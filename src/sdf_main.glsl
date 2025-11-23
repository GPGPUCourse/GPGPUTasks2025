#define PERIOD 5.0

#define CAMERA_POS vec3(fract(-iTime * 0.1) * PERIOD * 2.0, 5.0, 4.0)
#define CAMERA_DIR normalize(vec3(0.0, -1.0, -0.5))
#define FOV 120.0
#define UP normalize(vec3(0.0, 0.0, 1.0))

#define MARCH_MIN_T 1e-3
#define MARCH_MAX_T 1e5
#define MARCH_MIN_DIST 1e-5
#define MARCH_MAX_DIST 1e5
#define MARCH_ITER_COUNT 100
#define MARCH_SKY_TRIGGER 10.0

#define NORMAL_EPSILON 1e-3

#define SUN_DIR normalize(vec3(cos(iTime * 0.5), sin(iTime * 0.5), 0.5))

#define SHADOW_SOFTNESS 0.1

struct Ray {
    vec3 origin;
    vec3 direction;
};

vec3 fire(Ray ray, float t) {
    return ray.origin + ray.direction * t;
}

struct Material {
    vec3 albedo;
    float reflectiveness;
    float smoothness;
};

Material defaultMaterial = Material(
        vec3(0.0, 1.0, 1.0),
        0.0,
        1.0
    );

struct SDFResult {
    float dist;
    Material material;
};

// primitives

struct Sphere {
    vec3 position;
    float radius;
};

float sdf(Sphere s, vec3 pos) {
    return distance(pos, s.position) - s.radius;
}

struct Box {
    vec3 position;
    vec3 scale;
};

float sdf(Box b, vec3 pos) {
    vec3 dist = abs(pos - b.position) - b.scale;
    float max_dist = max(dist.x, max(dist.y, dist.z));
    return min(max_dist, 0.0) + length(max(dist, 0.0));
}

struct Plane {
    float level;
};

float sdf(Plane p, vec3 pos) {
    return pos.z - p.level;
}

struct Capsule {
    vec3 a;
    vec3 b;
    float radius;
};

float sdf(Capsule c, vec3 pos) {
    vec3 pa = pos - c.a, ba = c.b - c.a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - c.radius;
}

struct CappedTorus {
    float angle;
    float ra;
    float rb;
};

float sdf(CappedTorus t, vec3 pos) {
    pos.x = abs(pos.x);
    float k = (cos(t.angle) * pos.x > sin(t.angle) * pos.y) ?
        dot(pos.xy, vec2(sin(t.angle), cos(t.angle))) :
        length(pos.xy);
    return sqrt(dot(pos, pos) + t.ra * t.ra - 2.0 * t.ra * k) - t.rb;
}

// noise

float rand(vec3 v) {
    return fract(sin(dot(v, vec3(12.9898, 4.1414, 1.3434))) * 43758.5453);
}

float noise(vec3 v) {
    vec3 i = floor(v);
    vec3 f = fract(v);
    f = f * f * (3.0 - 2.0 * f);

    return mix(mix(mix(rand(i + vec3(0, 0, 0)),
                rand(i + vec3(1, 0, 0)), f.x),
            mix(rand(i + vec3(0, 1, 0)),
                rand(i + vec3(1, 1, 0)), f.x), f.y),
        mix(mix(rand(i + vec3(0, 0, 1)),
                rand(i + vec3(1, 0, 1)), f.x),
            mix(rand(i + vec3(0, 1, 1)),
                rand(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

const mat3 m = mat3(0.00, 0.80, 0.60,
        -0.80, 0.36, -0.48,
        -0.60, -0.48, 0.64);

float fbm(vec3 v) {
    vec3 q = 8.0 * v;
    float f = 0.5000 * noise(q);
    q = m * q * 2.01;
    f += 0.2500 * noise(q);
    q = m * q * 2.02;
    f += 0.1250 * noise(q);
    q = m * q * 2.03;
    f += 0.0625 * noise(q);
    q = m * q * 2.01;
    return f;
}

// scene

SDFResult sdfUnion(SDFResult a, SDFResult b) {
    if (a.dist < b.dist) {
        return a;
    }
    return b;
}

float smoothMin(float a, float b, float k) {
    k *= 4.0;
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

SDFResult ground(vec3 pos) {
    pos.xy = mod(pos.xy + PERIOD, vec2(2.0 * PERIOD)) - PERIOD;
    float tex = fbm(sin(pos * 3.1415 / PERIOD) * 0.2);

    return SDFResult(
        sdf(
            Plane(0.0),
            pos + vec3(0.0, 0.0, tex * 0.1)
        ),
        Material(
            vec3(mix(0.5, 1.0, tex), 0.0, 0.0),
            mix(0.0, 0.7, tex),
            0.999
        )
    );
}

SDFResult alien(vec3 pos) {
    vec3 origPos = pos;
    vec3 alienPos = vec3(0.0, fract(iTime * 0.1) * PERIOD * 2.0, 0.0);
    pos -= alienPos;
    pos.xy = mod(pos.xy + PERIOD, vec2(2.0 * PERIOD)) - PERIOD;
    float bounce1 = max(sin(iTime * 3.1415), 0.0);
    float bounce2 = max(sin((iTime + 1.0) * 3.1415), 0.0);
    vec3 bodyOffset = vec3(0.0, 0.0, (bounce1 + bounce2) * 0.1);
    vec3 bodyPos = pos - bodyOffset;
    float bottom = sdf(
            Sphere(
                vec3(0.0, 0.0, 1.5),
                1.0
            ),
            bodyPos
        );
    float top = sdf(
            Sphere(
                vec3(0.0, 0.0, 2.5),
                0.5
            ),
            bodyPos
        );
    float torso = smoothMin(bottom, top, 0.25);
    float limbs = min(
            min(
                sdf(Capsule(
                        vec3(0.8, 0.0, 2.0),
                        vec3(1.1 + sin(iTime * 10.0) * 0.1, 0.0, 2.8),
                        0.15
                    ), pos - bodyOffset),
                sdf(Capsule(
                        vec3(-0.8, 0.0, 2.0),
                        vec3(-1.2, 0.0, 1.2),
                        0.15
                    ), bodyPos)
            ),
            min(
                sdf(Capsule(
                        vec3(0.3, 0.0, 0.6) + bodyOffset,
                        vec3(0.3, 0.5 - abs(2.0 * fract(iTime * 0.5) - 1.0), 0.2 + 0.1 * bounce1),
                        0.2
                    ), pos),
                sdf(Capsule(
                        vec3(-0.3, 0.0, 0.6) + bodyOffset,
                        vec3(-0.3, 0.5 - abs(2.0 * fract((iTime + 1.0) * 0.5) - 1.0), 0.2 + 0.1 * bounce2),
                        0.2
                    ), pos)
            )
        );

    float mouth = sdf(CappedTorus(
                0.3,
                1.0,
                0.2
            ), vec3(bodyPos.x, 2.6 - bodyPos.z, bodyPos.y - 0.9));

    float tex = fbm(bodyPos);
    SDFResult body = SDFResult(
            smoothMin(-smoothMin(-torso, mouth, 0.01), limbs, 0.02),
            Material(
                vec3(0.0, mix(0.5, 1.0, tex), 0.0),
                mix(0.0, 0.3, tex),
                0.9
            )
        );

    SDFResult teeth = SDFResult(
            min(
                min(
                    sdf(Sphere(vec3(0.2, 0.7, 1.4), 0.15), bodyPos),
                    sdf(Sphere(vec3(-0.2, 0.7, 1.4), 0.15), bodyPos)
                ),
                min(
                    sdf(Sphere(vec3(0.2, 0.7, 1.85), 0.15), bodyPos),
                    sdf(Sphere(vec3(-0.2, 0.7, 1.85), 0.15), bodyPos)
                )
            ),
            Material(
                vec3(1.0),
                0.05,
                0.9
            )
        );

    vec3 eyePos = vec3(0.0, 0.5, 2.5);
    vec3 eyeLook = normalize(CAMERA_POS - eyePos + (bodyPos - origPos));
    vec3 eyeDir = normalize(bodyPos - eyePos);
    vec3 eyeCol = clamp(
            vec3(0.0, 0.0, 1.0) * step(dot(eyeDir, eyeLook), 0.95) +
                vec3(1.0) * step(dot(eyeDir, eyeLook), 0.8),
            vec3(0.0),
            vec3(1.0)
        );
    SDFResult eye = SDFResult(
            sdf(
                Sphere(
                    eyePos,
                    0.5
                ),
                bodyPos
            ),
            Material(
                eyeCol,
                0.1,
                1.0
            )
        );

    return sdfUnion(
        sdfUnion(
            body,
            teeth
        ),
        eye
    );
}

SDFResult sdf(vec3 pos) {
    return sdfUnion(
        alien(pos),
        ground(pos)
    );
}

// ray marching

struct MarchResult {
    float t;
    Material material;
    bool isSky;
};

MarchResult march(Ray ray) {
    MarchResult ans = MarchResult(MARCH_MIN_T, defaultMaterial, false);
    SDFResult res;
    for (int i = 0; i < MARCH_ITER_COUNT; i++) {
        res = sdf(fire(ray, ans.t));
        ans.material = res.material;
        ans.t += res.dist;
        if (res.dist < MARCH_MIN_DIST) {
            return ans;
        }
        if (res.dist > MARCH_MAX_DIST || ans.t > MARCH_MAX_T) {
            ans.isSky = true;
            return ans;
        }
    }
    ans.isSky = res.dist > MARCH_SKY_TRIGGER;
    return ans;
}

float softShadow(vec3 pos) {
    float shadow = 1.0;
    float t = MARCH_MIN_T;
    float cur_dist = 1.0;
    for (int i = 0;
        i < MARCH_ITER_COUNT &&
            t < MARCH_MAX_T &&
            shadow >= -1.0 &&
            cur_dist > MARCH_MIN_DIST;
        i++) {
        cur_dist = sdf(pos + SUN_DIR * t).dist;
        shadow = min(shadow, cur_dist / (SHADOW_SOFTNESS * t));
        t += clamp(cur_dist, 0.005, 0.50);
    }
    shadow = max(shadow, -1.0);
    return 0.25 * (1.0 + shadow) * (1.0 + shadow) * (2.0 - shadow);
}

vec3 calculateNormal(vec3 pos) {
    float center = sdf(pos).dist;
    vec3 delta = vec3(
            sdf(pos + vec3(NORMAL_EPSILON, 0.0, 0.0)).dist,
            sdf(pos + vec3(0.0, NORMAL_EPSILON, 0.0)).dist,
            sdf(pos + vec3(0.0, 0.0, NORMAL_EPSILON)).dist);
    return (delta - center) / NORMAL_EPSILON;
}

Ray reflection(Ray ray, MarchResult res) {
    ray.origin = fire(ray, res.t);
    vec3 normal = calculateNormal(ray.origin);
    vec3 random = normalize(vec3(
                rand(ray.origin),
                rand(ray.origin + 1.0),
                rand(ray.origin - 1.0)
            ) * 2.0 - 1.0);
    ray.direction = reflect(ray.direction, normal + random * (1.0 - res.material.smoothness));
    return ray;
}

Ray cameraRay(vec2 fragCoord) {
    vec2 uv = (fragCoord - iResolution.xy / 2.0) / iResolution.y;
    Ray camera = Ray(CAMERA_POS, CAMERA_DIR);
    vec3 planeCenter = fire(camera, tan(radians(90.0 - (FOV / 2.0))));
    vec3 planeRight = cross(CAMERA_DIR, UP);
    vec3 planeUp = cross(planeRight, CAMERA_DIR);
    vec3 planePoint = planeCenter + planeRight * uv.x + planeUp * uv.y;
    return Ray(CAMERA_POS, normalize(planePoint - CAMERA_POS));
}

vec3 sunColor = vec3(1.0, 1.0, 0.9);

vec3 skyColor(vec3 dir) {
    float sunStrength = clamp(mix(-2000.0, 10.0, dot(dir, SUN_DIR)), 0.0, 2.0);
    vec2 projected = dir.xy / dir.z;
    float clouds = fbm(vec3(projected * 0.5, 1.0)) * clamp(100.0 - length(projected), 0.0, 1.0);
    return mix(vec3(0.0, 0.0, 0.7), sunColor, sunStrength);
}

vec3 marchColor(Ray ray, MarchResult res) {
    return res.isSky ? skyColor(ray.direction) : res.material.albedo;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    Ray ray = cameraRay(fragCoord);
    float rayStrength = 1.0;
    vec3 col = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        MarchResult res = march(ray);
        if (res.isSky) {
            col += rayStrength * skyColor(ray.direction);
            break;
        }
        ray = reflection(ray, res);
        float shading = clamp(0.3 + softShadow(ray.origin), 0.0, 1.0);
        col += res.material.albedo * (shading * sunColor) *
                rayStrength * (1.0 - res.material.reflectiveness);
        rayStrength *= res.material.reflectiveness;
    }
    fragColor = vec4(col, 1.0);
}
