
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

float smin( float a, float b, float k )
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}

float opRound( in float p, in float rad )
{
    return p - rad;
}

vec4 sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return vec4(length( pa - ba*h ) - r, vec3(0.0, 1.0, 0.0));
}

vec4 opUnion(vec4 a, vec4 b) {
    return a.x < b.x ? a : b;
}


// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // TODO
    d = opRound(sdRoundCone((p - vec3(0.0, 0.5, -0.7)), 0.2, 0.1, 0.3), 0.18);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{

    vec3 color = vec3(1.0, 1.0, 1.0);

    p = p - vec3(0.0, 0.75, -0.45);
    if (length(p.xy) < 0.1) {
        color = vec3(0.35, 1, 1);
    }
    if (length(p.xy) < 0.06) {
        color = vec3(0.1, 0.1, 0.1);
    }

    if (length(p.xz) < cos(iTime)) {
        color = vec3(0.1, 1.0, 0.1);
    }

    vec4 res = vec4(sdSphere(p, 0.18), color);

    return res;
}

vec4 sdVerticalCapsule( vec3 p, float h, float r )
{
    p.y -= clamp( p.y, 0.0, h );
    return vec4(length( p ) - r, vec3(0.0, 1.0, 0.0));
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    vec4 res = sdBody(p);
    vec4 eye = sdEye(p);

    vec3 dir = vec3(-0.03, 0.3, 0.0) - vec3(0.1, 0.5, 0.0);
    float new_y = dir.y * cos(iTime) - dir.z * sin(iTime);
    float new_z = dir.y * sin(iTime) + dir.z * cos(iTime);
    dir.y = new_y;
    dir.z = new_z;
    vec4 r_arm = sdCapsule((p - vec3(-0.45, 0.0, -0.7)), vec3(0.1, 0.5, 0.0), vec3(0.1, 0.5, 0.0) + dir, 0.05);
    vec4 l_arm = sdCapsule((p - vec3(0.4, 0.0, -0.7)), vec3(-0.1, 0.5, 0.0), vec3(0.1, 0.5, 0.0) - dir, 0.05);
    vec4 l_leg = sdVerticalCapsule((p - vec3(0.2, 0.19, -0.7) - dir), 0.15 , 0.06);
    vec4 r_leg = sdVerticalCapsule((p - vec3(-0.2, 0.19, -0.7) + dir), 0.15, 0.06);

    res = opUnion(res, eye);
    res.x = smin(res.x, l_leg.x, 0.005);
    res.x = smin(res.x, r_leg.x, 0.005);
    res.x = smin(res.x, l_arm.x, 0.01);
    res.x = smin(res.x, r_arm.x, 0.01);

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