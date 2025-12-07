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

float smin( float a, float b, float k )
{
    k *= 6.0;
    float x = (b-a)/k;
    float g = (x> 1.0) ? x :
              (x<-1.0) ? 0.0 :
              (1.0+3.0*x*(x+1.0)-abs(x*x*x))/6.0;
    return b - k * g;
}


float sdLeg(vec3 p)
{
    float leg_base_d = sdSphere((p - vec3(0, 0.2, -0.7)), 0.05);

    float leg_d = sdSphere((p - vec3(0, -0.1, -0.7)), 0.03);
    
    leg_d = smin(leg_base_d, leg_d, 0.15);     
    
    return leg_d;
}

float sdArm(vec3 p, float dir)
{
    float t = iTime;

    float amp = 0.15;

    float angle = t * 3.0;
    float wave = lazycos(angle) * amp;

    vec3 offset = vec3(0.0, wave, 0.0) * dir;

    float arm_base_d = sdSphere((p - (vec3(0.4 * dir, 0.4, -0.7))), 0.05);

    float arm_d = sdSphere((p - (vec3(0.6 * dir, 0.45, -0.7) + offset)), 0.01);

    arm_d = smin(arm_base_d, arm_d, 0.1 + 0.019 * (1.0  - min(-dir * lazycos(angle), 0.05)));     
   
    return arm_d;
}



float opSmoothSubtraction( float d1, float d2, float k )
{

    k *= 4.0;
    float h = max(k-abs(-d1-d2),0.0);
    return max(-d1, d2) + h*h*0.25/k;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float body_d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.4);
    
    float head_d = sdSphere((p - vec3(0.0, 0.8, -0.7)), 0.3);

    float l_leg_d = sdLeg((p - vec3(-0.2, 0, 0)));

    float r_leg_d = sdLeg((p - vec3(0.2, 0, 0)));
    
    float l_arm_d = sdArm((p - vec3(0, 0, 0)), 1.0);

    float r_arm_d = sdArm((p - vec3(0, 0, 0)), -1.0);

    float d = smin(body_d, head_d, 0.08);

    d = smin(l_leg_d, d, 0.006);
    
    d = smin(r_leg_d, d, 0.006);
    
    d = smin(l_arm_d, d, 0.006);

    d = smin(r_arm_d, d, 0.006);

    
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    // lupoglaziy pedik
    float t = iTime;
    float r = 0.03;

    float angle = t * 1.5;

    vec2 circle = vec2(cos(angle), sin(angle)) * r;

    float white_d = sdSphere((p - vec3(0.0, 0.7, -0.45)), 0.2);

    float blue_d = sdSphere((p - vec3(circle.x, 0.7, -0.31 + circle.y)), 0.1);

    float black_d = sdSphere((p - vec3(circle.x * 1.2, 0.7, -0.26 + circle.y * 1.2)), 0.06);

    vec4 res;
    if (blue_d < white_d && blue_d < black_d) {
        res = vec4(blue_d, 0, 0, 1.0);
    } else if (white_d < blue_d && white_d < black_d) {
        res = vec4(white_d, 1.0, 1.0, 1.0);
    } else if (black_d < white_d && black_d < blue_d) {
        res = vec4(black_d, 0, 0, 0);
    }

    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.2, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    
    res.x = opSmoothSubtraction(eye.x, res.x, 0.01);
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

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3( 1, 0, 0,
                 0, c,-s,
                 0, s, c );
}

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(  c, 0, s,
                  0, 1, 0,
                 -s, 0, c );
}

mat3 rotZ(float a) {
    float c = cos(a), s = sin(a);
    return mat3(  c,-s, 0,
                  s, c, 0,
                  0, 0, 1 );
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
