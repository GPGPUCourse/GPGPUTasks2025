
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

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{
    float d_body = 1e10;
    float r_top = 0.3;
    float r_bot = 0.25;
    float height = 0.23;

    vec3 c = p - vec3(0.0, 0.35, -0.7);
    d_body = sdRoundCone(c, r_top, r_bot, height);

    float d = d_body;

    // hands
    float dx_hand = 0.1;
    float dy_hand = -0.12;
    float r_hand = 0.04;
    float a_x = 0.23, a_y = 0.1;

    // left hand
    vec3 a_lef = vec3(-a_x, a_y, 0.0);
    vec3 ab_lef = vec3(-dx_hand, lazycos(iTime * 6.0) * dy_hand, 0.0);
    float d_lef_hand = sdCapsule(c, a_lef, a_lef + ab_lef, r_hand);
    d = min(d, d_lef_hand);

    //right hand
    vec3 a_rig = vec3(a_x, a_y, 0.0);
    vec3 ab_rig = vec3(dx_hand, dy_hand, 0.0);
    float d_rig_hand = sdCapsule(c, a_rig, a_rig + ab_rig, r_hand);
    d = min(d, d_rig_hand);

    //legs
    float dy_leg = -0.42;
    float r_leg = 0.05;
    float a_x_leg = 0.1, a_y_leg = 0.1;
    vec3 ab_leg = vec3(0.0, dy_leg, 0.0);;

    // left hand
    vec3 a_lef_leg = vec3(-a_x_leg, a_y_leg, 0.0);
    float d_lef_leg = sdCapsule(c, a_lef_leg, a_lef_leg + ab_leg, r_leg);
    d = min(d, d_lef_leg);


    // left hand
    vec3 a_rig_leg = vec3(a_x_leg, a_y_leg, 0.0);
    float d_rig_leg = sdCapsule(c, a_rig_leg, a_rig_leg + ab_leg, r_leg);
    d = min(d, d_rig_leg);


    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec3 centr_1 = p - vec3(0.0, 0.59, -0.52);
    float r_1 = 0.15;
    float d_1 = sdSphere(centr_1, r_1);
    vec3 color_1 = vec3(1.0, 1.0, 1.0);

    vec4 res = vec4(d_1, color_1);

    vec3 d_centr_2 = vec3(0.0, 0.0, -0.01);
    float d_r2 = -0.0075;

    vec3 centr_2 = centr_1 + d_centr_2;
    float r_2 = r_1 + d_r2;
    float d_2 = sdSphere(centr_2, r_2);
    vec3 color_2 = vec3(0.0, 0.7, 1.0);

    if (d_2 < res.x) {
        res = vec4(d_2, color_2);
    }

    vec3 d_centr_3 = vec3(0.0, 0.0, -0.01);
    float d_r3 = -0.0092;

    vec3 centr_3 = centr_2 + d_centr_3;
    float r_3 = r_2 + d_r3;
    float d_3 = sdSphere(centr_3, r_3);
    vec3 color_3 = vec3(0.0, 0.0, 0.0);

    if (d_3 < res.x) {
        res = vec4(d_3, color_3);
    }

    return res;

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


vec4 raycast(vec3 ray_olefin, vec3 ray_direction)
{

    float EPS = 1e-3;


    // p = ray_olefin + t * ray_direction;

    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_olefin + t*ray_direction);
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


    vec3 ray_olefin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));


    vec4 res = raycast(ray_olefin, ray_direction);



    vec3 col = res.yzw;


    vec3 surface_point = ray_olefin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_olefin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;



    // Output to screen
    fragColor = vec4(col, 1.0);
}
