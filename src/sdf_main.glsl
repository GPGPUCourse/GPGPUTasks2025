#define PI 3.14159265

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }

mat2 rot(float a)
{
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

float opSmoothUnion( float d1, float d2, float k )
{
    k *= 4.0;
    float h = max(k - abs(d1 - d2),0.0);
    return min(d1, d2) - h * h * 0.25 / k;
}

float opSmoothSubtraction( float d1, float d2, float k )
{
    return -opSmoothUnion(d1,-d2,k);
}


float sinK(float k) {
    return sin(PI * (k - 0.5)) / 2.0 + 0.5;
}

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// elipsoid with center in (0, 0, 0)
float sdEllipsoid( vec3 p, vec3 r )
{
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float sdRoundCone( vec3 p, vec3 a, vec3 b, float r1, float r2 )
{
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;

    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    float x2 = dot2( pa*l2 - ba*y );
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // single square root!
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2>k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2<k ) return  sqrt(x2 + y2)        *il2 - r1;
    return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
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

float sdRightWavingArm(vec3 p, vec3 base, vec3 shoulder, float r1, float r2) {
    vec3 pr = p - shoulder;

    float idleTime  = 2.5;
    float riseTime  = 0.3;
    float waveTime  = 0.7;
    float fallTime  = 0.3;

    float cycle = idleTime + riseTime + waveTime + fallTime;
    float t = mod(iTime, cycle);

    float angle = 0.0;

    if (t > idleTime && t <= idleTime + riseTime)
    {
        float k = (t - idleTime) / riseTime;
        angle = mix(0.0, -2.0, sinK(k));
    }
    else if (t > idleTime + riseTime && t <= idleTime + riseTime + waveTime)
    {
        float k = (t - (idleTime + riseTime)) / waveTime;
        angle = mix(-2.0, -1.0, sinK(k * 4.0));
    }
    else if (t > idleTime + riseTime + waveTime)
    {
        float k = (t - (idleTime + riseTime + waveTime)) / fallTime;
        angle = mix(-2.0, 0.0, sinK(k));
    }

    pr.xy = rot(angle) * pr.xy;
    pr += shoulder;
    float d = sdRoundCone(pr, base, shoulder, 0.05, 0.08);
    return d;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // body
    float b1 = sdSphere(
        (p - vec3(0.0, 0.33, -0.7)),
        0.32
    );
    float b2 = sdEllipsoid(
        (p - vec3(0.0, 0.43, -0.7)),
        vec3(0.3, 0.4, 0.3)
    );
    float b3 = sdSphere(
        (p - vec3(0.0, 0.6, -0.5)),
        0.165
    );
    float b = opSmoothUnion(b1, b2, 0.018);
    b = opSmoothSubtraction(b3, b, 0.005);
    d = min(d, b);

    // legs
    float l1 = sdRoundCone(
        p,
        vec3(0.1, -0.03, -0.7),
        vec3(0.1, 0.15, -0.7),
        0.05,
        0.08
    );
    float l2 = sdRoundCone(
        p,
        vec3(-0.1, -0.03, -0.7),
        vec3(-0.1, 0.15, -0.7),
        0.05,
        0.08
    );
    d = min(d, min(l1, l2));

    // left arm
    float a1 = sdRoundCone(
        p,
        vec3(0.35, 0.25, -0.7),
        vec3(0.25, 0.38, -0.7),
        0.05,
        0.08
    );
    d = min(d, a1);

    // right waving arm
    float a2 = sdRightWavingArm(p, vec3(-0.35, 0.25, -0.7), vec3(-0.25, 0.38, -0.7), 0.05, 0.08);
    d = min(d, a2);

    // return distance and color
    return vec4(d, vec3(0.1, 1.0, 0.1));
}

vec4 sdEye(vec3 p)
{
    p -= vec3(0.0, 0.6, -0.5);
    float size = 0.165;

    float b1 = sdSphere(p, size);
    float b2 = sdSphere(p - vec3(0.0, 0.0, 0.05), size - 0.04);
    float b3 = sdSphere(p - vec3(0.0, 0.0, 0.085), size - 0.07);

    vec4 res = vec4(b1, 1.0, 1.0, 1.0);
    if (b2 < res.x) {
        res = vec4(b2, 0.0, 0.95, 0.95);
    }
    if (b3 < res.x) {
        res = vec4(b3, 0.0, 0.0, 0.0);
    }

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

    for (int iter = 0; iter < 500; ++iter) {
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

    // float t = iTime; // orbital rotating
    float t = 0.0; // for static

    float camAngle = sin(t * 0.4) * 1.5;
    float radius = 1.5;

    vec3 ray_origin = vec3(
    radius * sin(camAngle),
    0.4 + 0.2 * cos(camAngle),
    radius * cos(camAngle)
    );
    vec3 target = vec3(-1, 2, -0.7);

    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));
    ray_direction.xz = rot(camAngle) * ray_direction.xz;


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