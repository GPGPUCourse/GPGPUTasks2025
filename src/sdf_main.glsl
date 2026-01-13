
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p, vec3 n) {
    n = normalize(n);
    return dot(p,n) - 0.0;
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle) {
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
float smin(float a, float b, float k) {
    float h = clamp(0.5+0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

float sdRelativeCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float sdTorso(vec3 p, float horUpRadius, float vertUpRadius, float downRadius) {
    float up = sdEllipsoid(p, vec3(horUpRadius, vertUpRadius, horUpRadius));
    float down = sdSphere(p - vec3(0.0, -0.2, 0.0), downRadius);
    return smin(up, down, 0.15);
}

float sdVesicaSegment(in vec3 p, in vec3 a, in vec3 b, in float w) {
    vec3  c = (a+b)*0.5;
    float l = length(b-a);
    vec3  v = (b-a)/l;
    float y = dot(p-c,v);
    vec2  q = vec2(length(p-c-y*v),abs(y));

    float r = 0.5*l;
    float d = 0.5*(r*r-w*w)/w;
    vec3  h = (r*q.x<d*(q.y-r)) ? vec3(0.0,r,0.0) : vec3(-d,0.0,d+w);

    return length(q-h.xy) - h.z;
}

float sdCylinder(vec3 p, vec3 c) {
    return length(p.xz-c.xy)-c.z;
}

vec4 sdEyeColor(float bodyRadius, float d, float b, vec3 p, vec3 shift, vec3 bodyColor) {
    float bigRadius = 0.1;
    float smallRadius = 0.05;
    float eyeRadius = 0.15;

    vec3 begin = vec3(-bodyRadius * cos(b), 0.05, bodyRadius * sin(b));
    float eye = sdSphere(p - shift - begin, eyeRadius);
    if(d < eye) {
        return vec4(bodyColor, d);
    }

    // eye 1
    begin = vec3(-(bodyRadius + eyeRadius) * cos(b), 0.05, (bodyRadius + eyeRadius) * sin(b));
    float sphere = sdSphere(p - shift - begin, smallRadius);
    if(sphere < eye) {
        bodyColor = vec3(0.0);
        return vec4(bodyColor, eye);
    }

    // eye 2
    sphere = sdSphere(p - shift - begin, bigRadius);
    if(sphere < eye) {
        bodyColor = vec3(0.0, 1.0, 1.0);
        return vec4(bodyColor, eye);
    }

    return vec4(vec3(1.0), eye);
}

vec4 sdBody(vec3 p) {
    float d = 1e10;

    vec3 torsoPos = vec3(0.0, 0.5, -0.7);
    float turnLeftBy = -1.5 + 1.0*lazycos(3.0*iTime);
    float turnRightBy = -1.5 + 1.0*lazycos(2.0*iTime);
    float turnBodyBy = 0.5*sin(iTime) + 1.5f;

    // torso
    float horUpRadius = 0.3;
    float vertUpRadius = 0.4;
    float downRadius = 0.3;
    float bodyD = sdTorso(p - torsoPos, horUpRadius, vertUpRadius, downRadius);

    // ---- ARMS ----
    float h = 0.1;
    float b = turnBodyBy;

    // left arm
    float a = turnLeftBy;
    vec3 begin = vec3(downRadius * sin(b), 0.0, downRadius * cos(b));
    vec3 end = begin + vec3(
        +h * sin(b),
        -h * cos(a),
        +downRadius * sin(90.0 - b));
    float armLD = sdCapsule(p - torsoPos, begin, end, 0.04);

    // right arm
    a = turnRightBy;
    begin = vec3(-downRadius * sin(b), 0.0, -downRadius * cos(b));
    end = begin + vec3(
        -h * sin(b),
        -h * cos(a),
        -downRadius * sin(b - 90.0));
    float armRD = sdCapsule(p - torsoPos, begin, end, 0.04);

    // arms
    float arms = min(armLD, armRD);

    // ---- LEGS ----
    h = 0.2;
    float gamma = sin(3.0*iTime);
    a = 0.0;
    float legRadius = 0.15;

    // left arm
    begin = vec3(legRadius * sin(b), -torsoPos.y + h, legRadius * cos(b));
    end = begin + vec3(
        +0,
        -h * cos(a),
        +h * sin(gamma));
    float legLD = sdCapsule(p - torsoPos, begin, end, 0.04);

    // right arm
    gamma = -sin(3.0*iTime);
    begin = vec3(-legRadius * sin(b), -torsoPos.y + h, -legRadius * cos(b));
    end = begin + vec3(
        +0,
        -h * cos(a),
        +h * sin(gamma));
    float legRD = sdCapsule(p - torsoPos, begin, end, 0.04);

    // legs
    float legs = min(legLD, legRD);

    // connect bodyparts
    float allarmslegs = min(arms, legs);
    d = smin(bodyD, allarmslegs, 0.02);
    vec3 bodyColor = vec3(0.0, 1.0, 0.0);

    // FACE
    begin = vec3(-horUpRadius * cos(b), 0.05, horUpRadius * sin(b));
    vec4 eye = sdEyeColor(horUpRadius - 0.05, d, b, p, torsoPos, bodyColor);
    bodyColor = eye.xyz;
    d = smin(bodyD, eye.a, 0.02);

    return vec4(d, bodyColor);
}

vec4 sdEye(vec3 p) {
    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);

    return res;
}

vec4 sdMonster(vec3 p) {
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

vec4 sdTotal(vec3 p) {
    vec4 res = sdMonster(p);

    vec3 normal;
    normal.x = 0.05*lazycos(iTime);
    normal.y = 1.0 + 0.05*sin(iTime);
    normal.z = 0.05*sin(iTime);
    float dist = sdPlane(p, normal);

    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal(in vec3 p) {
    // for function f(p)
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize(vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
            sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
            sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x));
}

vec4 raycast(vec3 ray_origin, vec3 ray_direction) {
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

float shading(vec3 p, vec3 light_source, vec3 normal) {
    vec3 light_dir = normalize(light_source - p);

    float shading = dot(light_dir, normal);

    return clamp(shading, 0.5, 1.0);
}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness) {
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);

    return pow(max(dot(R, V), 0.0), shinyness);
}

float castShadow(vec3 p, vec3 light_source) {
    vec3 light_dir = p - light_source;

    float target_dist = length(light_dir);

    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
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