
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
float lazycos(float angle, int nsleep, int period)
{
    //int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < period) {
        return cos(angle);
    }

    return 1.0;
}

float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdArms(vec3 p) {
    vec3 c = vec3(0.0, 0.35, -0.7);

    float swing = 0.1 * (lazycos(iTime * 8.0, 10, 3) - 1.0);

    vec3 armL_a = c + vec3(0.3, 0.05, 0.0);
    vec3 armL_b = c + vec3(0.37, -0.05, 0.05);


    vec3 armR_a = c + vec3(-0.3, 0.05, 0.0);
    vec3 armR_b = c + vec3(-0.37, -0.05, 0.05);

    vec3 dirR = armR_b - armR_a;
    float lenR = length(dirR);
    dirR.y -= swing;
    dirR = normalize(dirR);
    armR_b = armR_a + dirR * lenR;

    float armL = sdCapsule(p, armL_a, armL_b, 0.05);
    float armR = sdCapsule(p, armR_a, armR_b, 0.05);

    return min(armL, armR);
}

float sdLegs(vec3 p) {
    vec3 c = vec3(0.0, 0.35, -0.7);

    vec3 legL_a = c + vec3(0.12, -0.10, 0.0);
    vec3 legL_b = c + vec3(0.12, -0.4, 0.05);

    vec3 legR_a = c + vec3(-0.12, -0.10, 0.0);
    vec3 legR_b = c + vec3(-0.12, -0.4, 0.05);

    float legL = sdCapsule(p, legL_a, legL_b, 0.065);
    float legR = sdCapsule(p, legR_a, legR_b, 0.065);

    return min(legL, legR);
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    float body = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);
    float head = sdSphere((p - vec3(0.0, 0.65, -0.7)), 0.2);

    float arms = sdArms(p);
    float legs = sdLegs(p);

    d = smin(body, head, 0.25);
    d = smin(d, arms, 0.02);
    d = smin(d, legs, 0.04);

    // return distance and color
    return vec4(d, vec3(0.0, 0.8, 0.0));
}

vec4 sdEye(vec3 p)
{

    vec3 c = vec3(0.0, 0.35, -0.7);
    vec3 e = c + vec3(0.0, 0.22, 0.2);

    float sclera = sdSphere(p - e, 0.18);
    float iris = sdSphere(p - (e + vec3(0.0, 0.004, 0.037)), 0.15);
    float pupil = sdSphere(p - (e + vec3(0.0, 0.01, 0.15)), 0.065);

    float d = sclera;
    vec3 col = vec3(1.0);

    if (iris < d) {
        d = iris;
        col = vec3(0.3, 0.9, 0.9);
    }

    if (pupil < d) {
        d = pupil;
        col = vec3(0.0);
    }

    float blink = lazycos(4.0 * iTime, 7, 1) - 1.0;

    vec3 ep = p - e;
    float lid_bottom = ep.y - (-0.2 - 0.11 * blink);
    float lid_top = -ep.y - (-0.2 - 0.11 * blink);
    float lid = min(lid_bottom, lid_top);
    float eyeLimit = sdSphere(ep - vec3(0.0, 0.0, 0.015), 0.18);
    lid = max(lid, eyeLimit);
    if (lid < sclera) {
        d = lid;
        col = vec3(0.0, 1.0, 0.0);
    }

    return vec4(d, col);
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