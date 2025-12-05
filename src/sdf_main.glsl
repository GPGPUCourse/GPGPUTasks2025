
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

// smooth union helper, see https://iquilezles.org/articles/smin/
float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// line segment with radius r
float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    // базовый силуэт — без колебаний, чтобы стоял неподвижно
    float wobble = 0.0;
    float main = sdSphere(p - vec3(0.0, 0.32 + wobble, -0.7), 0.33);
    float belly = sdSphere(p - vec3(0.0, 0.05, -0.6), 0.30);
    float head = sdSphere(p - vec3(0.0, 0.58 + wobble, -0.72), 0.22);

    float d = smin(main, belly, 0.12);
    d = smin(d, head, 0.10);

    // руки слегка двигаются постоянно
    float wave = 0.08 * sin(iTime * 1.5);
    float arm_left = sdCapsule(p, vec3(-0.30, 0.25, -0.6), vec3(-0.32 + wave, 0.42 + wave, -0.55), 0.055);
    float arm_right = sdCapsule(p, vec3(0.30, 0.25, -0.6), vec3(0.28 - wave, 0.42 + wave, -0.55), 0.055);
    d = smin(d, arm_left, 0.05);
    d = smin(d, arm_right, 0.05);

    // ножки
    float foot_left = sdCapsule(p, vec3(-0.12, -0.26, -0.55), vec3(-0.14, -0.36, -0.55), 0.10);
    float foot_right = sdCapsule(p, vec3(0.12, -0.26, -0.55), vec3(0.14, -0.36, -0.55), 0.10);
    d = smin(d, foot_left, 0.10);
    d = smin(d, foot_right, 0.10);

    // мягкие выпуклые бровки
    float brow_left = sdSphere(p - vec3(-0.12, 0.53 + 0.5 * wobble, -0.50), 0.09);
    float brow_right = sdSphere(p - vec3(0.12, 0.53 + 0.5 * wobble, -0.50), 0.09);
    d = smin(d, brow_left, 0.05);
    d = smin(d, brow_right, 0.05);

    // градиентный окрас: полоски по Z
    float stripes = 0.5 + 0.5 * sin(6.0 * p.z);
    vec3 base = vec3(0.06, 0.75, 0.16);
    vec3 highlight = vec3(0.18, 0.95, 0.28);
    vec3 col = mix(base, highlight, stripes * 0.7);

    // return distance and color
    return vec4(d, col);
}

vec4 sdEye(vec3 p)
{

    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);

    // плавное моргание: 1.0 — открыто, 0.35 — закрыто
    float blink = mix(0.35, 1.0, smoothstep(-0.2, 0.2, sin(iTime * 0.9)));

    // большой центральный глаз
    vec3 center = vec3(0.0, 0.50, -0.52);
    vec3 local = p - center;
    // сплющиваем по Y при моргании
    local.y *= blink;

    float eyeball = sdSphere(local, 0.16);
    vec3 color = vec3(0.94);

    float iris = sdSphere(local - vec3(0.0, -0.015, 0.08), 0.10);
    if (iris < eyeball) {
        eyeball = iris;
        color = vec3(0.16, 0.45, 0.75);
    }

    float pupil = sdSphere(local - vec3(0.0, -0.012, 0.11), 0.045);
    if (pupil < eyeball) {
        eyeball = pupil;
        color = vec3(0.02);
    }

    float sparkle = sdSphere(local - vec3(-0.05, 0.05, 0.07), 0.018);
    if (sparkle < eyeball) {
        eyeball = sparkle;
        color = vec3(1.0, 1.0, 0.9);
    }

    res = vec4(eyeball, color);

    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

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
