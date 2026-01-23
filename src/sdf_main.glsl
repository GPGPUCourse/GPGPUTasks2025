float dot2( in vec3 v ) { return dot(v,v); }

float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdRoundCone( vec3 p, float r1, float r2, float h )
{
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);

  vec2 q = vec2( length(p.xz), p.y );
  float k = dot(q,vec2(-b,a));
  if( k<0.0 ) return length(q) - r1;
  if( k>a*h ) return length(q-vec2(0.0,h)) - r2;
  return dot(q, vec2(a,b) ) - r1;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

float udRoundTriangle( vec3 p, vec3 a, vec3 b, vec3 c, in float rad )
{
  vec3 ba = b - a; vec3 pa = p - a;
  vec3 cb = c - b; vec3 pb = p - b;
  vec3 ac = a - c; vec3 pc = p - c;
  vec3 nor = cross( ba, ac );

  return sqrt(
    (sign(dot(cross(ba,nor),pa)) +
     sign(dot(cross(cb,nor),pb)) +
     sign(dot(cross(ac,nor),pc))<2.0)
     ?
     min( min(
     dot2(ba*clamp(dot(ba,pa)/dot2(ba),0.0,1.0)-pa),
     dot2(cb*clamp(dot(cb,pb)/dot2(cb),0.0,1.0)-pb) ),
     dot2(ac*clamp(dot(ac,pc)/dot2(ac),0.0,1.0)-pc) )
     :
     dot(nor,pa)*dot(nor,pa)/dot2(nor) ) - rad;
}

float sdPlane(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, 
// удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 2) {
        return cos(angle);
    }

    return cos(0.0);
}

float lazysin(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 2) {
        return sin(angle);
    }

    return sin(0.0);
}

vec3 rotate(vec3 p, float angle) 
{

    float c = cos(angle);
    float s = sin(angle);
    mat3 rotZ = mat3(
        c, -s, 0,
        s,  c, 0,
        0,  0, 1
    );
    
    return rotZ * p;
}

vec3 lazyrotate(vec3 p, float angle) 
{
    float c = abs(lazycos(angle * 0.7));
    float s = -abs(lazysin(angle * 0.7));
    mat3 rotZ = mat3(
        c, -s, 0,
        s,  c, 0,
        0,  0, 1
    );
    mat3 rotX = mat3(
        1,  0,  0,
        0,  c, -s,
        0,  s,  c
    );

    return rotZ * rotX * p;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора 
// https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: 
// https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = sdRoundCone(p, 0.30, 0.25, 0.35);
    return vec4(d, vec3(0.6, 0.2, 0.1));
}

vec4 sdTail(vec3 p, vec3 move, float angle)
{
    p += move;
    p = lazyrotate(p, angle);    
    float d = sdRoundCone(p, 0.04, 0.08, 0.45);
    return vec4(d, vec3(0.6, 0.2, 0.1));
}

vec4 sdEar(vec3 p, float h, vec3 move, float angle) 
{
    float width = 0.2;
    float height = 0.1;
    
    vec3 a = rotate(vec3(width / 2.0, h, 0), angle) + move;
    vec3 b = rotate(vec3(0.0, h + 0.1, 0.1), angle) + move;
    vec3 c = rotate(vec3(-width / 2.0, h, 0), angle) + move;
    
    float d = udRoundTriangle(p, a, b, c, 0.01);
    return vec4(d, vec3(0.6, 0.2, 0.1));
}


vec4 sdEye(vec3 p, vec3 move)
{
    p += move;
    float d = sdSphere(p, 0.02);
    return vec4(d, 0.1, 0.1, 0.1);
}

vec4 sdLeg(vec3 p, float radius, vec3 move, float angle)
{
    float height = 0.07;
    
    vec3 a = rotate(vec3(0.0, height / 2.0, 0.0), angle) + move;
    vec3 b = rotate(vec3(0.0, -height / 2.0, 0.0), angle) + move;
    
    float d = sdCapsule(p, a, b, radius);
    return vec4(d, vec3(0.6, 0.2, 0.1));
}

vec4 sdMouth(vec3 p, vec3 move)
{
    p += move;

    float radius = 0.01;
    float height = 0.02;
    
    vec3 a1 = rotate(vec3(0.0, height, 0.0), 0.0);
    vec3 b1 = rotate(vec3(0.0, -height, 0.0), -1.2);
    
    vec3 a2 = rotate(vec3(0.0, height, 0.0), 0.0);
    vec3 b2 = rotate(vec3(0.0, -height, 0.0), 1.2);
    
    float left = sdCapsule(p, a1, b1, radius);
    float right = sdCapsule(p, a2, b2, radius);
    
    float d = min(left, right);
    return vec4(d, vec3(0.1, 0.1, 0.1));
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p += vec3(0.0, -0.3, 0.7);

    vec4 res = sdBody(p);
    
    vec4 ear1 = sdEar(p, 0.6, vec3(0.2, 0, 0), -0.6);
    vec4 ear2 = sdEar(p, 0.6, vec3(-0.2, 0, 0), 0.6);

    vec4 eye1 = sdEye(p, vec3(0.1, -0.35, -0.24));
    vec4 eye2 = sdEye(p, vec3(-0.1, -0.35, -0.24));

    vec4 leg1 = sdLeg(p, 0.08, vec3(-0.12, -0.18, 0.17), 0.18);
    vec4 leg2 = sdLeg(p, 0.08, vec3(0.12, -0.18, 0.17), -0.18);

    vec4 leg3 = sdLeg(p, 0.1, vec3(-0.22, -0.18, 0.0), 1.2);
    vec4 leg4 = sdLeg(p, 0.1, vec3(0.22, -0.18, 0.0), -1.2);
    
    vec4 tail = sdTail(p, vec3(-0.1, 0.05, 0.0), iTime);
    vec4 mouth = sdMouth(p, vec3(0.0, -0.3, -0.25));
    
    vec4 resEye = eye1.x < eye2.x ? eye1 : eye2;
    vec4 resEar = ear1.x < ear2.x ? ear1 : ear2;
    vec4 resLegFront = leg1.x < leg2.x ? leg1 : leg2;
    vec4 resLegBack = leg3.x < leg4.x ? leg3 : leg4;
    
    res = tail.x < res.x ? tail : res;
    res = mouth.x < res.x ? mouth : res;
    res = resEye.x < res.x ? resEye : res;
    res = resEar.x < res.x ? resEar : res;
    res = resLegFront.x < res.x ? resLegFront : res;
    res = resLegBack.x < res.x ? resLegBack : res;
    
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
