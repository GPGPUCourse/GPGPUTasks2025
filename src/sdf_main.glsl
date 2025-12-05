
const vec3 center = vec3(0.0, 0.35, -0.7);
const float delta_x_hand = 0.1;
const float delta_y_hand = -0.1;
const float radius_hand = 0.05;
const float hand_x = 0.3;
const float hand_y = 0.1;

// sphere with center in (0, 0, 0)
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
    float d = 1e10;

    // TODO
    d = sdRoundCone((p - center), 0.35, 0.27, 0.35);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    float d = 1e10;

    // TODO
    float y = 0.7;
    float z = -0.5;
    float x = 0.0;
    d = sdSphere((p - vec3(x, y, z)), 0.17);
    if (p.z > z) {
        if (pow(abs(p.y - y), 2.0) + pow(abs(p.x - x), 2.0) < 0.003) {
            return vec4(d, vec3(0.0, 0.0, 0.0));
        }
        if (pow(abs(p.y - y), 2.0) + pow(abs(p.x - x), 2.0) < 0.01) {
            return vec4(d, vec3(0.188, 0.835, 0.784));
        }
    }
    // return distance and color
    return vec4(d, vec3(1.0, 1.0, 1.0));
}

vec4 sdLeftHand(vec3 p)
{
    float d = 1e10;
    vec3 hand = vec3(-hand_x, hand_y, 0.0);
    vec3 delta_hand = vec3(-delta_x_hand, lazycos(iTime * 7.0) * delta_y_hand, 0.0);
    d = sdCapsule(p - center, hand, hand + delta_hand, radius_hand);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdRightHand(vec3 p)
{
    float d = 1e10;
    vec3 hand = vec3(hand_x, hand_y, 0.0);
    vec3 delta_hand = vec3(delta_x_hand,  delta_y_hand, 0.0);
    d = sdCapsule(p - center, hand, hand + delta_hand, radius_hand);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdLeftLeg(vec3 p) {
    float d = sdVerticalCapsule(p - center - vec3(-0.2, -0.38, 0.0), 0.15, radius_hand);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdRightLeg(vec3 p) {
    float d = sdVerticalCapsule(p - center - vec3(0.2, -0.38, 0.0), 0.15, radius_hand);
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
    vec4 leftHand = sdLeftHand(p);
    if (leftHand.x < res.x) {
        res = leftHand;
    }
    
    vec4 rightHand = sdRightHand(p);
    if (rightHand.x < res.x) {
        res = rightHand;
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
