
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

float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5*(b - a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0 - h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d1 = sdSphere(p - vec3(0.0, 0.25, -0.7), 0.35);

    float d2 = sdSphere((p - vec3(0.0, 0.45, -0.7)) * vec3(1.0, 1.3, 1.0), 0.32);

    float d = smin(d1, d2, 0.25);

    float armSwing = 0.25 * lazycos(iTime * 3.0);

    vec3 armL_a = vec3(-0.27, 0.32, -0.7);
    vec3 armL_b = vec3(-0.45, 0.32 + armSwing, -0.7);
    float armL = sdCapsule(p, armL_a, armL_b, 0.07);

    vec3 armR_a = vec3(0.27, 0.32, -0.7);
    vec3 armR_b = vec3(0.45, 0.32, -0.7);
    float armR = sdCapsule(p, armR_a, armR_b, 0.07);

    d = smin(d, armL, 0.03);
    d = smin(d, armR, 0.03);

    vec3 legL_a = vec3(-0.13, 0.10, -0.63);
    vec3 legL_b = vec3(-0.13, -0.25, -0.63);
    float legL = sdCapsule(p, legL_a, legL_b, 0.12);

    vec3 legR_a = vec3(0.13, 0.10, -0.63);
    vec3 legR_b = vec3(0.13, -0.25, -0.63);
    float legR = sdCapsule(p, legR_a, legR_b, 0.12);

    d = min(d, legL);
    d = min(d, legR);

    return vec4(d, 0.0, 1.0, 0.0);
}

vec4 sdEye(vec3 p)
{
    vec3 E = vec3(0.0, 0.47, -0.45);

    float Rwhite = 0.18;
    float Riris  = 0.09;
    float Rpupil = 0.05;

    float dWhite = sdSphere(p - E, Rwhite);
    float dIris  = sdSphere(p - (E + vec3(0.0, 0.0, 0.17)), Riris);
    float dPupil = sdSphere(p - (E + vec3(0.0, 0.0, 0.25)), Rpupil);

    float d = dWhite;
    vec3 col = vec3(1.0);

    if (dIris  < d) { d = dIris;  col = vec3(0.2, 0.5, 1.0); }
    if (dPupil < d) { d = dPupil; col = vec3(0.05); }

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