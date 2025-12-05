#define PI 3.14159265

// sigmoid
float smin( float a, float b, float k )
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
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

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / (2. * PI)) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

vec4 sdBody(vec3 p)
{
    float d = smin(
        sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.34),
        sdSphere((p - vec3(0.0, 0.85, -0.7)), 0.1),
        0.19
    );

    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdIris(vec3 p) {
    float d1 = sdSphere((p - vec3(0.0, 0.71, -0.343)), 0.1);
    float d2 = sdSphere((p - vec3(0.0, 0.71, -0.4)), 0.15);
    
    if (d1 < d2) {
        return vec4(d1, vec3(0.0));
    }
    return vec4(d2, vec3(0.0, 1.0, 0.898));
}

vec4 sdEye(vec3 p)
{

    float d1 = sdSphere((p - vec3(0.0, 0.71, -0.46)), 0.2);
    
    return vec4(d1, vec3(1.0, 1.0, 1.0)); 
}

vec4 sdLimb(vec3 p, vec3 p1, vec3 p2) {
    float d = sdCapsule(p, p1, p1 + p2, 0.055);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 nextMin(vec4 cur, vec4 new) {
    if (new.x < cur.x) {
        return new;
    }
    return cur;
}

float handMove(float t) {
    t -= 5.0;
    float T = 4.0;
    float f = 0.0;
    f += sin(2.0 * PI * 1.0 * t / T) / 1.0;
    f += sin(2.0 * PI * 3.0 * t / T) / 3.0;
    f += sin(2.0 * PI * 5.0 * t / T) / 5.0;
    f += sin(2.0 * PI * 7.0 * t / T) / 7.0;
    f *= (4.0 / 3.14159265);
    return max(f, -0.9);
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, -.2);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    res = nextMin(res, eye);
    
    vec4 iris = sdIris(p);
    res = nextMin(res, iris);
    
    vec3 leftHandOr = vec3(-.34, .4, -.6);
    vec3 leftHandDelta = vec3(-0.09, 0.12 / 0.9 * handMove(iTime), 0.0);
    vec4 leftHand = sdLimb(p, leftHandOr, leftHandDelta);
    res = nextMin(res, leftHand);
    
    vec3 rightHandOr = vec3(.34, .4, -.6);
    vec3 rightHandDelta = vec3(0.07, -0.12, 0.0);
    vec4 rightHand = sdLimb(p, rightHandOr, rightHandDelta);
    res = nextMin(res, rightHand);
    
    vec3 leftLegOr = vec3(-.14, .1, -.6);
    vec3 leftLegDelta = vec3(0.0, -0.12, 0.0);
    vec4 leftLeg = sdLimb(p, leftLegOr, leftLegDelta);
    res = nextMin(res, leftLeg);
    
    vec3 rightLegOr = vec3(.14, .1, -.6);;
    vec3 rightLegDelta = vec3(0.0, -0.12, 0.0);
    vec4 rightLeg = sdLimb(p, rightLegOr, rightLegDelta);
    res = nextMin(res, rightLeg);
    
    
    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(246, 177, 206) / 170.);
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

    return vec4(1e10, vec3(0.2, 0.5, 1.8) * 3.2);
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{

    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.47, .92);

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

    vec3 light_source = vec3(1.0 + 4.*sin(iTime), 10.5, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 50.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}
