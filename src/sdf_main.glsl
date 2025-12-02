
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

float sUnion( float d1, float d2, float k )
{
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

vec4 minByX(vec4 a, vec4 b) {
    return (b.x < a.x) ? b : a;
}

vec3 rotate(vec3 p, vec3 axis, vec3 center, float angle) {
    p -= center;
    axis = normalize(axis);
    float ax = axis.x * angle;
    float ay = axis.y * angle;
    float az = axis.z * angle;
    mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cos(ax), -sin(ax), 0.0, sin(ax), cos(ax));
    mat3 ry = mat3(cos(ay), 0.0, sin(ay), 0.0, 1.0, 0.0, -sin(ay), 0.0, cos(ay));
    mat3 rz = mat3(cos(az), -sin(az), 0.0, sin(az), cos(az), 0.0, 0.0, 0.0, 1.0);
    return rx * ry * rz * p + center;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    vec3 center = vec3(0.0, 0.35, -0.7);
    float r1 = 0.32;
    float r2 = 0.25;
    float h = 0.3;
    
    d = sUnion(sdSphere(p - center, r1), sdSphere(p - center + vec3(0.0, -h, 0.0), r2), 0.22);
    
    // return distance and color
    return vec4(d, vec3(0.2, 0.4, 1.0));
}

vec4 sdEye(vec3 p)
{
    vec3 center = vec3(0.0, 0.6, -0.52);
    float r = 0.2;
    float d = sdSphere((p - center), r);
    
    vec3 eyeDir = vec3(0.0, 0.2, 1.0);
    float angle = max(0.0, dot(normalize(p - center), normalize(eyeDir)));
    
    vec3 color = vec3(1.0, 1.0, 1.0);
    if (angle > 0.95) {
        color = vec3(0.0, 0.0, 0.0);
    } 
    else if (angle > 0.85) {
        color = vec3(1.0, 0.7, 0.4);
    }

    return vec4(d, color);
}

vec4 sdMouth(vec3 p)
{
    vec3 face = vec3(0.0, 0.6, -0.32);
    vec3 a = face + vec3(-0.10, -0.05, 0.02);
    vec3 b = face + vec3( 0.10, -0.05, 0.02);
    float r = 0.035;

    float d = sdCapsule(p, a, b, r);

    return vec4(d, vec3(1.0, 1.0, 1.0));
}


vec4 sdLimbs(vec3 p) {
    vec3 c1 = vec3(-0.1, 0.05, -0.7);
    vec3 c2 = vec3(0.1, 0.05, -0.7);
    vec3 c3 = vec3(-0.3, 0.4, -0.7);
    vec3 c4 = vec3(0.3, 0.4, -0.7);
    
    vec3 legDir = vec3(0.0, -0.08, 0.0);
    vec3 armDir1 = vec3(-0.15, -0.08, 0.2);
    armDir1 = rotate(armDir1, vec3(-1.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), lazycos(iTime * 8.0));
    vec3 armDir2 = vec3(0.15, -0.08, 0.0);
    vec3 color = vec3(0.2, 0.4, 1.0);
    
    vec4 res = vec4(sdCapsule(p, c1, c1 + legDir, 0.05), color);
    res = minByX(res, vec4(sdCapsule(p, c2, c2 + legDir, 0.05), color));
    res = minByX(res, vec4(sdCapsule(p, c3, c3 + armDir1, 0.05), color));
    res = minByX(res, vec4(sdCapsule(p, c4, c4 + armDir2, 0.05), color));
    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 res = sdBody(p);
    res = minByX(res, sdEye(p));
    res = minByX(res, sdMouth(p + vec3(0.0, 0.3, 0.0)));
    res = minByX(res, sdLimbs(p));

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.2, 0.1));
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
