
// Rotation matrix around the X axis.
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

// Rotation matrix around the Y axis.
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

// Rotation matrix around the Z axis.
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

// Identity matrix.
mat3 identity() {
    return mat3(
        vec3(1, 0, 0),
        vec3(0, 1, 0),
        vec3(0, 0, 1)
    );
}

float sdCappedCone( vec3 p, float h, float r1, float r2 )
{
  vec2 q = vec2( length(p.xz), p.y );
  vec2 k1 = vec2(r2,h);
  vec2 k2 = vec2(r2-r1,2.0*h);
  vec2 ca = vec2(q.x-min(q.x,(q.y<0.0)?r1:r2), abs(q.y)-h);
  vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2, k2), 0.0, 1.0 );
  float s = (cb.x<0.0 && ca.y<0.0) ? -1.0 : 1.0;
  return s*sqrt( min(dot(ca, ca),dot(cb, cb)) );
}

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// quadratic polynomial
vec2 smin_color( float a, float b, float k )
{
    float h = 1.0 - min( abs(a-b)/(4.0*k), 1.0 );
    float w = h*h;
    float m = w*0.5;
    float s = w*k;
    return (a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m);
}

float sigmoid_min( float a, float b, float k )
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}

float sdEllipsoid( vec3 p, vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
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
    float top_d = 1e10;
    float main_d = 1e10;

    top_d = sdRoundCone((p - vec3(0.0, 0.4, -0.7)), 0.35, 0.3, 0.3);
    main_d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);

    // return distance and color
    return vec4(sigmoid_min(top_d, main_d, 0.1), vec3(0.0, 1.0, 0.0));
}

vec4 sdArm(vec3 p, vec3 shift, mat3 rotation)
{
    float d = 1e10;
    vec3 center = (p + shift - vec3(0.0, 0.35, -0.5));
    d = sdEllipsoid(center * rotation, vec3(0.2, 0.05, 0.05));

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdLeg(vec3 p, vec3 shift)
{
    float d = 1e10;
    d = sdEllipsoid((p + shift - vec3(0.0, 0.0, -0.5)), vec3(0.05, 0.2, 0.05));

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdHat(vec3 p)
{
    float bottom_d = 1e10;
    float main_d = 1e10;

    bottom_d = sdEllipsoid((p - vec3(0.0, 0.9, -0.6)), vec3(0.4, 0.01, 0.4));
    main_d = sdCappedCone((p - vec3(0.0, 0.9, -0.6)), 0.2, 0.09, 0.2);

    // return distance and color
    return vec4(sigmoid_min(bottom_d, main_d, 0.1), vec3(0.45, 0.32, 0.1));
}


vec4 sdEye(vec3 p, vec3 shift)
{
    float main_d = 1e10;
    float small_d = 1e10;
    float very_small_d = 1e10;
    float eps = 1e-2;

    main_d = sdEllipsoid((p + shift - vec3(0.0, 0.55, -0.4)), vec3(0.11, 0.16, 0.1));
    small_d = sdEllipsoid((p + shift - vec3(0.0, 0.55, -0.35)), vec3(0.05, 0.09, 0.06));
    very_small_d = sdEllipsoid((p + shift - vec3(0.0, 0.554, -0.30)), vec3(0.01, 0.02, 0.01));
    vec2 first_res = smin_color(main_d, small_d, 0.01);
    vec2 second_res = smin_color(first_res.x, very_small_d, 0.001);
    vec3 first_col = mix( vec3(1.0,1.0,1.0), vec3(0.2, 0.2, 1.0), first_res.y );
    vec3 second_col = mix( first_col, vec3(0.0, 0.0, 0.0), second_res.y );
    if (abs(main_d - second_res.x) < eps) {
        second_col = vec3(1.0, 1.0, 1.0);
    }
    if (abs(small_d - second_res.x) < eps) {
        second_col = vec3(0.2, 0.5, 1.0);
    }
    if (abs(very_small_d - second_res.x) < eps) {
        second_col = vec3(0.0, 0.0, 0.0);
    }
    return vec4(second_res.x, second_col.x, second_col.y, second_col.z);
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.2, -0.4);

    vec4 res = sdBody(p);

    vec4 right_eye = sdEye(p, vec3(-0.15, 0.0, 0.0));
    if (right_eye.x < res.x) {
        res = right_eye;
    }

    vec4 left_eye = sdEye(p, vec3(0.15, 0.0, 0.0));
    if (left_eye.x < res.x) {
        res = left_eye;
    }
    float time = iTime;
    if (int(floor(time / 1.07)) % 2 == 1) {
        time = 1.07 - mod(time, 1.07);
    }
    else {
        time = mod(time, 1.07);
    }
    float angle = time * 1.5 - 3.14 / 4.0;

    vec4 left_arm = sdArm(p, vec3(0.4, 0.0, 0.0), rotateZ(angle));
    if (left_arm.x < res.x) {
        res = left_arm;
    }

    vec4 right_arm = sdArm(p, vec3(-0.4, 0.0, 0.0), rotateZ(3.14 / 4.0));
    if (right_arm.x < res.x) {
        res = right_arm;
    }

    vec4 left_leg = sdLeg(p, vec3(0.15, 0.0, 0.0));
    if (left_leg.x < res.x) {
        res = left_leg;
    }

    vec4 right_leg = sdLeg(p, vec3(-0.15, 0.0, 0.0));
    if (right_leg.x < res.x) {
        res = right_leg;
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