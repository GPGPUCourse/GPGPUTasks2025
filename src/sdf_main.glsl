
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

float smin( float a, float b, float k )
{
    k *= log(2.0);
    float x = b-a;
    return a + x/(1.0-exp2(x/k));
}


float sdCylinder(vec3 a, vec3 b, float r, vec3 p)
{
    vec3 ba = b - a;
    float h = length(ba);
    ba = normalize(ba);
    vec3 pa = p - a;
    float t = dot(pa, ba);
    t = clamp(t, 0.0, h);
    vec3 q = a + t * ba;
    vec3 pq = p - q;
    float d = length(pq);
    return d - r;
}

float sdCapsule(vec3 a, vec3 b, float r, vec3 p)
{
    float topSphere = sdSphere(p - a, r);
    float res = topSphere;
    float bottomSphere = sdSphere(p - b, r);
    if (res > bottomSphere) {
        res = bottomSphere;
    }
    float cylinder = sdCylinder(a, b, r, p);
    if (res > cylinder) {
        res = cylinder;
    }
    return res;
}

float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

vec4 sdBody(vec3 p)
{

    float d1 = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.4);
    float d2 = sdSphere((p - vec3(0.0, 0.7, -0.7)), 0.3);

    // return distance and color
    return vec4(smin(d1, d2, 0.05), vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec4 res = vec4(sdSphere(p, 0.16), vec3(1.0, 1.0, 1.0));
    p -= vec3(0.0, 0.0, 0.1);
    vec4 zr = vec4(sdSphere(p, 0.08), vec3(0.0, 0.0, 1.0));
    if (res.x > zr.x) {
        res = zr;
    }
    p -= vec3(0.0, -0.025, 0.1);
    vec4 hole = vec4(sdSphere(p, 0.03), vec3(0.0, 0.0, 0.0));
    if (res.x > hole.x) {
        res = hole;
    }   
    return res;
}

vec4 sdMonster(vec3 p)
{
    p -= vec3(0, 0.08, 0.0);

    vec4 res = sdBody(p - vec3(0.0, 0.2, 0.0));
    vec4 eye = sdEye(p + vec3(0.0, -0.88, 0.47));
    if (eye.x < res.x) {
        res = eye;
    }
    vec4 rightLeg = vec4(sdCapsule(vec3(0.15, 0.0, -0.7), vec3(0.1, 0.45, -0.7), 0.06, p), vec3(0.0, 1.0, 0.0));
    if (rightLeg.x < res.x) {
        res = rightLeg;
    }
    vec4 leftLeg = vec4(sdCapsule(vec3(-0.15, 0.0, -0.7), vec3(-0.1, 0.45, -0.7), 0.06, p), vec3(0.0, 1.0, 0.0));
    if (leftLeg.x < res.x) {
        res = leftLeg;
    }
    vec4 rightArm = vec4(sdCapsule(vec3(0.28, 0.7, -0.55), vec3(0.5, 0.4, -0.6), 0.06, p), vec3(0.0, 1.0, 0.0));
    if (rightArm.x < res.x) {
        res = rightArm;
    }
    vec4 leftArm = vec4(sdCapsule(vec3(-0.28, 0.7, -0.55), vec3(-0.5, 0.4, -0.6), 0.06, p), vec3(0.0, 1.0, 0.0));
    if (leftArm.x < res.x) {
        res = leftArm;
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