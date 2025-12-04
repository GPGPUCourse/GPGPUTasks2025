
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

float dot2(in vec3 v ) { return dot(v,v); }
float sdRoundCone( vec3 p, vec3 a, vec3 b, float r1, float r2 )
{
  vec3  ba = b - a;
  float l2 = dot(ba,ba);
  float rr = r1 - r2;
  float a2 = l2 - rr*rr;
  float il2 = 1.0/l2;
    
  vec3 pa = p - a;
  float y = dot(pa,ba);
  float z = y - l2;
  float x2 = dot2( pa*l2 - ba*y );
  float y2 = y*y*l2;
  float z2 = z*z*l2;

  // single square root!
  float k = sign(rr)*rr*rr*x2;
  if( sign(z)*a2*z2>k ) return  sqrt(x2 + z2)        *il2 - r2;
  if( sign(y)*a2*y2<k ) return  sqrt(x2 + y2)        *il2 - r1;
                        return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

float opSmoothUnion( float d1, float d2, float k )
{
    k *= 4.0;
    float h = max(k-abs(d1-d2),0.0);
    return min(d1, d2) - h*h*0.25/k;
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
    float k = 0.025;

    float torso = sdRoundCone(
        (p - vec3(0.0, 0.5, -0.7)),
        vec3(0, 0.2, 0),
        vec3(0, -0.1, 0),
        0.27f,
        0.35f
        );

    float leftLeg = sdRoundCone(
        (p - vec3(-0.1, 0.1, -0.7)),
        vec3(0, 0.1, 0),
        vec3(0, -0.1, 0),
        0.05f,
        0.05f
        );
        
    float rightLeg = sdRoundCone(
        (p - vec3(0.1, 0.1, -0.7)),
        vec3(0, 0.1, 0),
        vec3(0, -0.1, 0),
        0.05f,
        0.05f
        );
    
    float legs = min(leftLeg, rightLeg);
    float torsoWithLegs = opSmoothUnion(torso, legs, k);
    
    float angle = iTime * 6.28318530718 * 3.0;
    float myCos = lazycos(angle);
    float mySin = sqrt(1.0 - myCos * myCos);
    
    float leftArm = sdRoundCone(
        (p - vec3(-0.35, 0.3, -0.7)),
        vec3(0, 0.1, 0),
        vec3(abs(mySin) * -0.2, myCos * -0.2 + 0.1, 0),
        0.05f,
        0.05f
        );
        
    float rightArm = sdRoundCone(
        (p - vec3(0.35, 0.3, -0.7)),
        vec3(0, 0.1, 0),
        vec3(0, -0.1, 0),
        0.05f,
        0.05f
        );
        
    float arms = min(leftArm, rightArm);
    float torsoWithArms = opSmoothUnion(torsoWithLegs, arms, k);
    

    // return distance and color
    return vec4(torsoWithArms, vec3(1.0, 0.8, 0.0));
}

vec4 sdEye(vec3 p)
{
    float whiteEye = sdSphere(p + vec3(0.0, -0.6, 0.6), 0.25);
    float blueEye = sdSphere(p + vec3(0.0, -0.65, 0.41), 0.075);

    if (whiteEye < blueEye) {
        return vec4(whiteEye, 1.0, 1.0, 1.0);
    } else {
        return vec4(blueEye, 0.0, 0.0, 1.0);
    }
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
    
    #if 1
    float d = 2.3;
    
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec4 curr = sdMonster(p + vec3(float(i) * d, 0.0, float(j) * d));
            if (curr.x < res.x) res = curr;
        }
    }
    #endif


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

    return vec4(1e10, vec3(0.46328125, 0.53984375, 0.9375));
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
    float sum_weight = 1.0;
    
    #if 1
    vec3 refract_point = ray_origin + res.x*ray_direction;
    vec3 refract_normal = calcNormal(refract_point);
    
    for (int i = 1; i <= 10; i++) {
        vec4 refract_res = raycast(refract_point, refract_normal);
        float weight = 1.0 / float(i) / 1.0;
        col += weight * refract_res.yzw;
        sum_weight += weight;
        if (refract_point.x > 1E5) break;
        refract_point = refract_point + refract_res.x * refract_normal;
        refract_normal = calcNormal(refract_point);
    }
    col /= sum_weight;
    #endif

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