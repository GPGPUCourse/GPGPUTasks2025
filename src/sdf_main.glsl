
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    vec3 center = vec3(0.0, 0.38, -0.7);
    vec3 radii = vec3(0.28, 0.38, 0.28);
    vec3 q = p - center;
    float t = clamp((q.y + radii.y) / (2.0 * radii.y), 0.0, 1.0);
    float scale = mix(0.99, 1.0, t);
    q.xz *= scale;
    float d = sdEllipsoid(q, radii);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec3 center = vec3(0.0, 0.52, -0.45);
    vec4 res = vec4(sdSphere(p - center, 0.09), vec3(1.0, 1.0, 1.0));
    vec3 irisP = p - vec3(0.0, 0.0, 0.15);
    vec4 iris = vec4(sdSphere(irisP - center, 0.06), vec3(0.2, 0.7, 1.0));
    if ( iris.x < res.x ) {
        res = iris;
    }
    vec3 pupilP = p - vec3(0.0, 0.0, 0.2);
    vec4 pupil = vec4(sdSphere(pupilP - center, 0.03), vec3(0.0, 0.0, 0.0));
    if ( pupil.x < res.x ) {
        res = pupil;
    }
    return res;
}

//руки/ноги
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

vec4 sminSDF(vec4 a, vec4 b, float k) {
    float h = clamp(0.5 + 0.5 * (b.x - a.x) / k, 0.0, 1.0);
    float d = mix(b.x, a.x, h) - k * h * (1.0 - h);
    return (a.x < b.x) ? vec4(d, a.yzw) : vec4(d, b.yzw);
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 body = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x > body.x) {
        eye = vec4(1e10, 0.0, 0.0, 0.0);
    }
    vec4 res = sminSDF(body, eye, 0.04);

    // --- РУКИ ---
    float handAnim = 0.5 + 0.5 * lazycos(4.0 * iTime);
    vec3 leftA = vec3(-0.23, 0.38, -0.7);
    vec3 leftB = leftA + vec3(-0.13, 0.13 * handAnim, 0.0); 
    float leftHand = sdCapsule(p, leftA, leftB, 0.045);

    vec3 rightA = vec3(0.23, 0.38, -0.7);
    vec3 rightB = rightA + vec3(0.13, 0.13, 0.0);
    float rightHand = sdCapsule(p, rightA, rightB, 0.045);

    vec3 handColor = vec3(0.0, 1.0, 0.0);
    vec4 leftHandSDF = vec4(leftHand, handColor);
    vec4 rightHandSDF = vec4(rightHand, handColor);
    res = sminSDF(res, leftHandSDF, 0.03);
    res = sminSDF(res, rightHandSDF, 0.03);

    // --- НОГИ ---
    vec3 leftLegA = vec3(-0.10, 0.08, -0.7);
    vec3 leftLegB = leftLegA + vec3(0.0, -0.18, 0.0);
    float leftLeg = sdCapsule(p, leftLegA, leftLegB, 0.055);
    vec3 rightLegA = vec3(0.10, 0.08, -0.7);
    vec3 rightLegB = rightLegA + vec3(0.0, -0.18, 0.0);
    float rightLeg = sdCapsule(p, rightLegA, rightLegB, 0.055);
    vec3 legColor = vec3(0.0, 1.0, 0.0);
    vec4 leftLegSDF = vec4(leftLeg, legColor);
    vec4 rightLegSDF = vec4(rightLeg, legColor);
    res = sminSDF(res, leftLegSDF, 0.03);
    res = sminSDF(res, rightLegSDF, 0.03);

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
    sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
    sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{

    float EPS = 1e-3;


    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.0, 0.0, 0.0));
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{

    vec3 light_dir = p - light_source;

    float target_dist = length(light_dir);


    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));


    vec4 res = raycast(ray_origin, ray_direction);



    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}