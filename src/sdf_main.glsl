
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

float sdRoundCone(vec3 p, float r1, float r2, float h)
{
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);

  vec2 q = vec2( length(p.xz), p.y );
  float k = dot(q,vec2(-b,a));
  if( k<0.0 ) return length(q) - r1;
  if( k>a*h ) return length(q-vec2(0.0,h)) - r2;
  return dot(q, vec2(a,b) ) - r1;
}

float opSmoothUnion(float d1, float d2, float k)
{
    k *= 4.0;
    float h = max(k-abs(d1-d2),0.0);
    return min(d1, d2) - h*h*0.25/k;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float bot = sdSphere(p - vec3(0.0, 0.3, -0.7), 0.3);
    float top = sdSphere(p - vec3(0.0, 0.7, -0.7), 0.1);

    float d = opSmoothUnion(bot, top, 0.15);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    float d = 1e10;

    p = p - vec3(0.0, 0.605, -0.48);
    d = sdSphere(p, 0.16);

    vec3 color = vec3(1.0,1.0,1.0);
    float r = p.x * p.x + p.y *p.y;
    if (r < 0.0085) color = vec3(0.0, 1.0, 1.0);
    if (r < 0.0025) color = vec3(0.0, 0.0, 0.0);

    return vec4(d, color);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

vec4 sdLeftArm(vec3 p)
{
    float t = iTime * 3.0;
    float c = lazycos(t); 
    float s = sqrt(1.0 - c * c);

    vec3 a = vec3(0.0, s * 0.1, 0.0);
    vec3 b = vec3(0.05, 0.1, 0.0);
    float d = sdCapsule(p - vec3(-0.32, 0.3, -0.52), a, b, 0.05);

    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdRightArm(vec3 p)
{
    vec3 a = vec3(0.05, 0.0, 0.0);
    vec3 b = vec3(0.0, 0.1, 0.0);
    float d = sdCapsule(p - vec3(0.27, 0.3, -0.52), a, b, 0.05);

    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdLeftLeg(vec3 p) {
    vec3 a = vec3(0.0, 0.0, 0.0);
    vec3 b = vec3(0.0, 0.1, 0.0);
    float d = sdCapsule(p - vec3(-0.1, 0.0, -0.55), a, b, 0.07);

    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdRightLeg(vec3 p) {
    vec3 a = vec3(0.0, 0.0, 0.0);
    vec3 b = vec3(0.0, 0.1, 0.0);
    float d = sdCapsule(p - vec3(0.1, 0.0, -0.55), a, b, 0.07);

    return vec4(d, vec3(0.0, 1.0, 0.0));
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

    vec4 rightArm = sdRightArm(p);
    if (rightArm.x < res.x) {
        res = rightArm;
    }

    vec4 leftArm = sdLeftArm(p);
    if (leftArm.x < res.x) {
        res = leftArm;
    }

    vec4 leftLeg = sdLeftLeg(p);
    if (leftLeg.x < res.x) {
        res = leftLeg;
    }

    vec4 rightLeg = sdRightLeg(p);
    if (rightLeg.x < res.x) {
        res = rightLeg;
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
