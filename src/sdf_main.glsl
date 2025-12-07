
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }


float sdCappedCone(vec3 p, float h, float r1, float r2)
{
  vec2 q = vec2(length(p.xz), p.y);
  vec2 k1 = vec2(r2, h);
  vec2 k2 = vec2(r2 - r1,2.0 * h);
  vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
  vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot2(k2), 0.0, 1.0);
  float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
  return s * sqrt(min(dot2(ca), dot2(cb)));
}

float sdRoundCone( vec3 p, float h, float r1, float r2)
{
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);

  vec2 q = vec2( length(p.xz), p.y );
  float k = dot(q,vec2(-b,a));
  if( k<0.0 ) return length(q) - r1;
  if( k>a*h ) return length(q-vec2(0.0,h)) - r2;
  return dot(q, vec2(a,b) ) - r1;
}

float sdVerticalCapsule( vec3 p, float h, float r )
{
  p.y -= clamp( p.y, 0.0, h );
  return length( p ) - r;
}

float sdOctahedron( vec3 p, float s )
{
  p = abs(p);
  float m = p.x+p.y+p.z-s;
  vec3 q;
       if( 3.0*p.x < m ) q = p.xyz;
  else if( 3.0*p.y < m ) q = p.yzx;
  else if( 3.0*p.z < m ) q = p.zxy;
  else return m*0.57735027;
    
  float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
  return length(vec3(q.x,q.y-s+k,q.z-k)); 
}

float sdTorus(vec3 p, vec2 t)
{
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 1;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

// sigmoid
float smin(float a, float b, float k)
{
    k *= log(2.0);
    float x = b - a;
    return a + x / (1.0 - exp2(x / k));
}

vec3 sdRotate(vec3 p, float cosine) {
    return vec3(p.x * cosine + p.y * sqrt(1.0 - cosine * cosine), p.y * cosine - p.x * sqrt(1.0 - cosine * cosine), p.z);
}

vec4 posoh_deda_moroza(vec3 p, float cosine) {
    float d1 = 1e10, d2 = 1e10, d3 = 1e10;
    
    vec2 t = vec2(0.03, 0.01);
    
    vec3 rot_p = sdRotate(p - vec3(-0.3, 0.45, -0.5), cosine * 0.5);
    rot_p = sdRotate(rot_p, -0.5);
    
    d1 = sdVerticalCapsule((rot_p - vec3(0.2, -0.9, 0.0)), 0.9, 0.02);
    d2 = sdOctahedron((rot_p - vec3(0.2, -0.9, 0.0)), 0.1);
    d3 = sdTorus((rot_p - vec3(0.2, -0.81, 0.0)), t);
    
    if (smin(d1, d2, 0.01) < d3) {
        return vec4(smin(d1, d2, 0.01), vec3(1.0, 1.0, 1.0));
    }
    return vec4(d3, vec3(1.0, 0.0, 0.0));
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d1 = 1e10, d2 = 1e10, d3 = 1e10, d4 = 1e10, d5 = 1e10, d6 = 1e10, d7 = 1e10;
    
    float cosine = lazycos(iTime);
    vec3 rot_p_left = sdRotate(p - vec3(-0.3, 0.45, -0.5), cosine * 0.5);
    vec3 rot_p_right = sdRotate(p - vec3(0.45, 0.35, -0.5), 0.5);

    d1 = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.4);
    d2 = sdSphere((p - vec3(0.0, 0.7, -0.7)), 0.3);

    d3 = sdRoundCone(rot_p_left, 0.2, 0.04, 0.07);
    d4 = sdRoundCone(rot_p_right, 0.2, 0.07, 0.04);
    
    d5 = sdRoundCone((p - vec3(-0.2, -0.1, -0.7)), 0.2, 0.07, 0.04);
    d6 = sdRoundCone((p - vec3(0.2, -0.1, -0.7)), 0.2, 0.07, 0.04);
    
    float min_result = smin(d1, d3, 0.02);
    min_result = smin(min_result, d4, 0.01);
    min_result = smin(min_result, d2, 0.06);
    min_result = smin(min_result, d5, 0.01);
    min_result = smin(min_result, d6, 0.01);
    
    vec4 posoh = posoh_deda_moroza(p, cosine);
    
    if (posoh.x < min_result) {
        return posoh;
    }
    return vec4(min_result, vec3(0.0, 1.0, 0.0));
}

vec4 sdHat(vec3 p)
{
    float d1 = 1e10, d2 = 1e10, d3 = 1e10;
    
    vec2 t = vec2(0.3, 0.05);

    d1 = sdCappedCone((p - vec3(0.0, 1.15, -0.7)), 0.25, 0.3, 0.05);
    d2 = sdTorus((p - vec3(0.0, 0.89, -0.7)), t);
    d3 = sdSphere((p - vec3(0.0, 1.45, -0.7)), 0.08);
    
    if (d1 < d2) {
        if (d1 < d3) {
            return vec4(d1, vec3(1.0, 0.0, 0.0));
        }
        return vec4(d3, vec3(1.0, 1.0, 1.0));
    }
    return vec4(d2, vec3(1.0, 1.0, 1.0));
}

vec4 sdEye(vec3 p)
{
    float d1 = 1e10, d2 = 1e10, d3 = 1e10;

    d1 = sdSphere((p - vec3(0.0, 0.65, -0.5)), 0.2);
    d2 = sdSphere((p - vec3(0.0, 0.67, -0.37)), 0.1);
    d3 = sdSphere((p - vec3(0.0, 0.67, -0.3)), 0.05);
    
    if (d1 < d2) {
        return vec4(d1, vec3(1.0, 1.0, 1.0));
    }
    if (d2 < d3) {
       return vec4(d2, vec3(1.0, 0.2, 0.5));
    }
    return vec4(d3, vec3(0.0, 0.0, 0.0));
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.2, -1.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }
    
    vec4 hat = sdHat(p);
    if (hat.x < res.x) {
        res = hat;
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
    vec2 uv = fragCoord / iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);


    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5 * wh, -1.0));


    vec4 res = raycast(ray_origin, ray_direction);


    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0, 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;


    // Output to screen
    fragColor = vec4(col, 1.0);
}