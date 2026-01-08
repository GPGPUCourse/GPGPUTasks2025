// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

vec3 rotateZ(vec3 p, float a) {
    float c = cos(a);
    float s = sin(a);
    return vec3(
        c*p.x - s*p.y,
        s*p.x + c*p.y,
        p.z
    );
}

float smin( float a, float b, float k )
{
    k *= 1.0;
    float r = exp2(-a/k) + exp2(-b/k);
    return -k*log2(r);
}


float linInterpolation(float x, float from, float to) {
    x = (x + 1.0);
    x = (to - from) * x / 2.0 + from;
    return x;
}

float sdVerticalCapsule( vec3 p, float h, float r )
{
  p.y -= clamp( p.y, 0.0, h );
  return length( p ) - r;
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

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = sdVerticalCapsule((p - vec3(0.0, 0.35, 0.0)), 0.4, 0.3);    
    
    float tol = 0.02;
    
    { // right hand
        vec3 _center = vec3(0.2, 0.6, -0.1);
        vec3 _p = rotateZ(p - _center, 3.14 * linInterpolation(cos(iTime * 5.0), 0.3, 0.5));
    
        float _d = sdVerticalCapsule(_p, 0.3, 0.05);
        d = smin(_d, d, tol);
    }
    
    { // left hand
        vec3 _center = vec3(-0.2, 0.6, 0.1);
        vec3 _p = rotateZ(p - _center, -2.5);
    
        float _d = sdVerticalCapsule(_p, 0.3, 0.05);
        d = smin(_d, d, tol);
    }
    
    { // right leg
        vec3 _center = vec3(0.1, 0.2, 0.1);
        vec3 _p = rotateZ(p - _center, 2.5);
    
        float _d = sdVerticalCapsule(_p, 0.3, 0.05);
        d = smin(_d, d, tol);
    }
    
    { // left leg
        vec3 _center = vec3(-0.1, 0.2, 0.1);
        vec3 _p = rotateZ(p - _center, -2.5);
    
        float _d = sdVerticalCapsule(_p, 0.3, 0.05);
        d = smin(_d, d, tol);
    }

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    float d = sdSphere((p - vec3(0.0, 0.7, 0.25)), 0.1);
    
    {
        float _d = sdSphere((p - vec3(0.0, 0.7, 0.35)), 0.05 * linInterpolation(cos(iTime), 0.5, 1.0));
        if(_d < d) return vec4(d, 0.0, 0.0, 0.0);
    }
    
    return vec4(d, 1.0, 1.0, 1.0);
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.0, 0.0);

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
    
    float R = 3.5;
    float ang = iTime * 0.5;

    vec3 target = vec3(0.0, 0.3, 0.0);

    vec3 ray_origin = vec3(
        R * cos(ang),
        1.5,
        R * sin(ang)
    );

    vec3 forward = normalize(target - ray_origin);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);

    vec2 p = uv - 0.5 * wh;
    vec3 ray_direction = normalize(
        p.x * right +
        p.y * up +
        1.5 * forward
    );

    vec4 res = raycast(ray_origin, ray_direction);

    vec3 col = res.yzw;

    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(8.5, 10.0, 7.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}