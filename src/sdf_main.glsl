
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


float sdVerticalCapsule( vec3 p, float h, float r )
{
  p.y -= clamp( p.y, 0.0, h );
  return length( p ) - r;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
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
float lazysin(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

float smin( float a, float b, float k )
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d1 = 1e10, d2;

    // TODO
    d1 = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.3);
    d2 = sdSphere((p - vec3(0.0, 0.6, -0.7)), 0.22);

    // return distance and color
    return vec4(smin(d1, d2, 0.045), vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    float d = sdSphere((p - vec3(0.0, 0.55, -0.47)), 0.15);
    vec3 c = vec3(1.0, 1.0, 1.0);
    if (length(p - vec3(0.0, 0.55, -0.32)) < 0.1) {
        c = vec3(0.13, 0.5, 0.9);
    }
    if (length(p - vec3(0.0, 0.55, -0.32)) < 0.05) {
        c = vec3(0.0, 0.0, 0.0);
    }

    return vec4(d, c);
}


vec4 sdLegs(vec3 p) {
    float dLeft = sdVerticalCapsule(p - vec3(0.1, 0, -0.7), 0.2, 0.05);
    float dRight = sdVerticalCapsule(p - vec3(-0.1, 0, -0.7), 0.2, 0.05);
    return vec4(min(dLeft, dRight), vec3(0.0, 1.0, 0.0));
}

vec4 sdLeftArm(vec3 p) {
    float d = sdCapsule(p, vec3(0.29, 0.35, -0.6), vec3(0.35, 0.25, -0.5), 0.025);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdRightArm(vec3 p) {
    vec3 pT = vec3(-0.29, 0.35, -0.6);
    vec3 pW = vec3(-0.44, 0.35, -0.6);
    float len = length(pT - pW);
    float dy = lazycos(3.0*iTime) * 0.8;
    pW.y = pT.y - dy * len;
    pW.x = pT.x - sqrt(1.0-dy*dy) * len;
    float d = sdCapsule(p, pT, pW, 0.025);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 body = sdBody(p);
    vec4 res = body;

    vec4 legs = sdLegs(p);
    vec4 leftArm = sdLeftArm(p);
    vec4 rightArm = sdRightArm(p);
    res.x = smin(body.x, smin(legs.x, leftArm.x, 0.01), 0.01);
    res.x = smin(res.x, rightArm.x, 0.008);
    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }
    if (res.x > smin(body.x, eye.x, 0.01)) {
    //    res.x = smin(body.x, eye.x, 0.01);
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

