
float opSmoothUnion( float d1, float d2, float k )
{
    k *= 6.0;
    float h = max(k-abs(d1-d2),0.0);
    return min(d1, d2) - h*h*0.25/k;
}

float opSmoothSubtraction( float d1, float d2, float k )
{
    return -opSmoothUnion(d1,-d2,k);

    //k *= 4.0;
    // float h = max(k-abs(-d1-d2),0.0);
    // return max(-d1, d2) + h*h*0.25/k;
}

float opSmoothIntersection( float d1, float d2, float k )
{
    return -opSmoothUnion(-d1,-d2,k);

    //k *= 4.0;
    // float h = max(k-abs(d1-d2),0.0);
    // return max(d1, d2) + h*h*0.25/k;
}

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

float opUnion( float d1, float d2 )
{
    return min(d1,d2);
}

float opSubtraction( float d1, float d2 )
{
    return max(-d1,d2);
}

float opIntersection( float d1, float d2 )
{
    return max(d1,d2);
}

float opXor( float d1, float d2 )
{
    return max(min(d1,d2),-max(d1,d2));
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 5;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

// left/right leg
float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

// head
float sdDeathStar( vec3 p2, float ra, float rb, float d )
{
  float a = (ra*ra - rb*rb + d*d)/(2.0*d);
  float b = sqrt(max(ra*ra-a*a,0.0));

  vec2 p = vec2( p2.x, length(p2.yz) );
  if( p.x*b-p.y*a > d*max(b-p.y,0.0) )
    return length(p-vec2(a,b));
  else
    return max( (length(p            )-ra),
               -(length(p-vec2(d,0.0))-rb));
}

float sdEllipsoid( vec3 p, vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;
    { // left
        vec3 low = vec3(0.0, -0.3, 0.0);
        vec3 up = vec3(0.0, -0.2, 0.0);
        d = sdCapsule((p - vec3(0.0, 0.35, -0.7)), low, up, 0.05);
    }
    
    { // right
        vec3 low = vec3(0.15, -0.3, 0.0);
        vec3 up = vec3(0.15, -0.2, 0.0);
        d = opUnion(d, sdCapsule((p - vec3(0.0, 0.35, -0.7)), low, up, 0.05));
    }
    
    { // left arm
        float t = iTime * 30.0;
        float x = lazycos(t) * 0.013 - 0.16;
        float y = sqrt(0.1 - x * x) - 0.24;
        vec3 low = vec3(x, y, 0.0);
        vec3 up = vec3(-0.08, -0.12, 0.0);
        d = opUnion(d, sdCapsule((p - vec3(0.0, 0.35, -0.7)), low, up, 0.02));
    }
    
    { // right arm
        
        vec3 low = vec3(0.222, -0.25, 0.0);
        vec3 up = vec3(0.222, -0.12, 0.0);
        d = opUnion(d, sdCapsule((p - vec3(0.0, 0.35, -0.7)), low, up, 0.02));
    }
    
    
    { // head
        vec3 r = vec3(0.17, 0.1, 0.1);
        float d_body = 1e10;
        d_body = opUnion(d_body, sdEllipsoid((p - vec3(0.07, 0.23, -0.7)), r));
        r = vec3(0.15, 0.2, 0.1);
        d_body = opSmoothUnion(d_body, sdEllipsoid((p - vec3(0.07, 0.33, -0.7)), r), 0.01);
        d = opUnion(d, d_body);
    }
    
        
    // return distance and color
    return vec4(d, vec3(0.5, 1.5, 0.5));
}

vec4 sdEye(vec3 p)
{
    float d = 1e10;
    d = sdSphere((p - vec3(0.06, 0.39, -0.60)), 0.1);
    vec3 color = vec3(1.0, 1.0, 1.0);
    float norm = p.x * p.x + 0.5 * p.z * p.z;
    if (norm < 0.14) {
        color = vec3(0.0, 0.0, 2.0);
    }
    if (norm < 0.13) {
        color = vec3(0.2, 0.2, 0.2);
    }

    return vec4(d, color);
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