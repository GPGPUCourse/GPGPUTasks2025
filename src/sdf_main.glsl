float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
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
    float dHead = sdSphere(p - vec3(0.0, 0.55, 0.0), 0.32);
    float dButt = sdSphere(p - vec3(0.0, 0.30, 0.0), 0.35);
    float d = smin(dHead, dButt, 0.15);

    vec3 pLegs = p;
    pLegs.x = abs(pLegs.x);
    float legs = sdCapsule(pLegs, vec3(0.14, 0.2, 0.05), vec3(0.14, -0.05, 0.05), 0.08);
    d = smin(d, legs, 0.01);

    float speed = 4.0; 
    float cycleLen = 32.0; 
    float t = mod(iTime * speed, cycleLen);
    
    float animL = 0.0;
    float animR = 0.0;
    
    vec3 bodyColor = vec3(0.0, 0.8, 0.1); 

    if (t < 24.0) {
        float localT = mod(t, 8.0);
        
        animL = smoothstep(0.0, 1.0, localT) - smoothstep(6.0, 7.0, localT);
        animR = smoothstep(2.0, 3.0, localT) - smoothstep(4.0, 5.0, localT);
    } else {
        float localT = t - 24.0;
        
        float pump = sin(localT * 3.14159 / 2.0); 
        pump = pump * pump; 
        
        animL = pump;
        animR = pump;
        
        vec3 shift = vec3(3.14, 0.93, 2.50);
        
        float freq = 3.14159 * 0.5; 
        
        bodyColor = 0.5 + 0.5 * cos(freq * localT + shift);
    }

    vec3 shoulderL = vec3(-0.32, 0.5, 0.0);
    vec3 shoulderR = vec3( 0.32, 0.5, 0.0);
    vec3 dirDown = normalize(vec3(0.5, -0.85, 0.15));
    vec3 dirUp   = normalize(vec3(0.8,  0.60, 0.10));
    
    vec3 dirDownL = dirDown * vec3(-1.0, 1.0, 1.0);
    vec3 dirUpL   = dirUp   * vec3(-1.0, 1.0, 1.0);
    
    vec3 handL = shoulderL + mix(dirDownL, dirUpL, animL) * 0.22;
    float armL = sdCapsule(p, shoulderL, handL, 0.06);
    
    vec3 handR = shoulderR + mix(dirDown, dirUp, animR) * 0.22;
    float armR = sdCapsule(p, shoulderR, handR, 0.06);
    
    d = smin(d, min(armL, armR), 0.04);

    return vec4(d, bodyColor); 
}

vec4 sdEye(vec3 p)
{
    vec3 center = vec3(0.0, 0.55, 0.25);
    
    float dWhite = sdSphere(p - center, 0.19);
    vec4 res = vec4(dWhite, vec3(1.0, 1.0, 1.0)); 
    
    float dIris = sdSphere(p - (center + vec3(0.0, 0.0, 0.13)), 0.1);
    if (dIris < res.x) {
        res = vec4(dIris, vec3(0.0, 0.4, 1.0)); 
    }
    
    float dPupil = sdSphere(p - (center + vec3(0.0, 0.0, 0.19)), 0.055);
    if (dPupil < res.x) {
        res = vec4(dPupil, vec3(0.0, 0.0, 0.0)); 
    }

    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    vec3 q = p - vec3(0.0, 0.1, -0.5); 

    vec4 body = sdBody(q);
    
    vec4 eye = sdEye(q);

    if (eye.x < body.x) {
        return eye;
    }
    return body;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);


    float dist = sdPlane(p);
    if (dist < res.x) {
        float cells = mod(floor(p.x * 0.5 + 0.15) + floor(p.z * 0.5 + 0.15), 2.0);
        
        vec3 floorCol = mix(vec3(0.4, 0.0, 0.0), vec3(0.8, 0.1, 0.1), cells);
        
        res = vec4(dist, floorCol);
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


    float speed = 4.0; 
    float cycleLen = 32.0;
    float t = mod(iTime * speed, cycleLen);

    float camZ = 1.0;
    
    if (t > 24.0) {
        float localT = t - 24.0;
        
        float fastPump = sin(localT * 3.14159 * 0.5);
        
        fastPump = fastPump * fastPump; 
        
        camZ = 1.0 - (fastPump * 0.05);
    }
    
    vec3 ray_origin = vec3(0.0, 0.5, camZ);
    
    float shakeX = (sin(iTime * 2.3) + cos(iTime * 2.1)) * 0.5;
    float shakeY = (cos(iTime * 7.7) + sin(iTime * 7.8)) * 0.8;
    
    ray_origin.xy += vec2(shakeX, shakeY) * 0.015;
    
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