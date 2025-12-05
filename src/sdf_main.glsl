float smin(float a, float b, float k) {
    k *= 16.0 / 3.0;
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * h * (4.0 - h) * k * (1.0 / 16.0);
}

vec4 sminVec4(vec4 a, vec4 b, float k) {
    float h = clamp(0.5 + 0.5 * (b.x - a.x) / k, 0.0, 1.0);
    float d = mix(b.x, a.x, h) - k * h * (1.0 - h);
    vec3 col = mix(b.yzw, a.yzw, h);
    return vec4(d, col);
}

vec3 getLightPos() {
    return vec3(1.0 + 2.5 * sin(iTime), 10.0, 10.0);
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle) {
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if(iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

float dot2(in vec3 v) {
    return dot(v, v);
}

////////////////////////////////////////////////////
// Примитивы 

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p) {
    return p.y;
}

float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 ba = b - a;
    float l2 = dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    float il2 = 1.0 / l2;

    vec3 pa = p - a;
    float y = dot(pa, ba);
    float z = y - l2;
    float x2 = dot2(pa * l2 - ba * y);
    float y2 = y * y * l2;
    float z2 = z * z * l2;

  // single square root!
    float k = sign(rr) * rr * rr * x2;
    if(sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if(sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

float sdCutHollowSphere(vec3 p, float r, float h, float t) {
    float w = sqrt(r * r - h * h);
    vec2 q = vec2(length(p.xz), p.y);
    return ((h * q.x > w * q.y) ? length(q - vec2(w, h)) : abs(length(q) - r)) - t;
}

float sdVerticalCapsule(vec3 p, float h, float r) {
    p.y -= clamp(p.y, 0.0, h);
    return length(p) - r;
}

float sdRhombus(vec3 p, float la, float lb, float h, float ra) {
    p = abs(p);
    float f = clamp((la * p.x - lb * p.z + lb * lb) / (la * la + lb * lb), 0.0, 1.0);
    vec2 w = p.xz - vec2(la, lb) * vec2(f, 1.0 - f);
    vec2 q = vec2(length(w) * sign(w.x) - ra, p.y - h);
    return min(max(q.x, q.y), 0.0) + length(max(q, 0.0));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdCappedTorus(vec3 p, vec2 sc, float ra, float rb) {
    p.x = abs(p.x);
    float k = (sc.y * p.x > sc.x * p.y) ? dot(p.xy, sc) : length(p.xy);
    return sqrt(dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
}
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

////////////////////////////////////////////////////
// Тело 

vec4 sdPupil(vec3 p, vec3 eyeCenter) {
    vec3 light_pos = getLightPos();
    vec3 dirToCam = normalize(light_pos - eyeCenter + vec3(0.0, -10.0, 0.0));

    float offset = 0.15;
    float r = 0.082;

    vec3 center = eyeCenter + dirToCam * offset;

    float d = sdSphere(p - center, r);

    return vec4(d, vec3(0.02, 0.02, 0.02));
}

vec4 sdIris(vec3 p, vec3 eyeCenter) {
    vec3 light_pos = getLightPos();
    vec3 dirToCam = normalize(light_pos - eyeCenter + vec3(0.0, -10.0, 0.0));

    float offset = 0.13;
    float r = 0.1;

    vec3 center = eyeCenter + dirToCam * offset;

    float d = sdSphere(p - center, r);

    return vec4(d, vec3(0.17, 0.65, 0.65));
}

vec4 sdUpperEyelid(vec3 p, vec3 eyeCenter) {
    // радиус века 
    float R = 0.23;

    float h_base = 0.085;
    float blinkSpeed = 1.0;
    float blink = 0.5 - 0.5 * cos(iTime * 3.14159 * blinkSpeed);
    float h = h_base * (1.0 - blink); 

    // толщина века
    float t = 0.01;

    vec3 local = p - eyeCenter;

    float d = sdCutHollowSphere(local, R, h, t);

    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p) {
    vec3 eyeCenter = vec3(0.0, 0.65, -0.51);
    float eyeRadius = 0.20;

    vec4 dEye = vec4(sdSphere(p - eyeCenter, 0.20), vec3(0.999, 0.99, 0.98));
    vec4 pupil = sdPupil(p, eyeCenter);
    vec4 iris = sdIris(p, eyeCenter);
    vec4 lid = sdUpperEyelid(p, eyeCenter);

    const float EPS = 0.0007;

    vec4 temp = sminVec4(dEye, iris, EPS);
    temp = sminVec4(temp, pupil, EPS);
    temp = sminVec4(temp, lid, EPS);

    return temp;
}

vec4 sdLeg(vec3 p) {

    float d = 1e10;
    float d2 = 1e10;

    d = sdVerticalCapsule(p - vec3(0.17, 0.0, -0.8), 0.3, 0.05);
    d2 = sdRhombus(p - vec3(0.17, -0.045, -0.63), 0.037, 0.2, 0.025, 0.03);

    const float k = 0.03;

    return vec4(smin(d, d2, k), vec3(0.0, 1.0, 0.0));
}

vec4 sdLegPair(vec3 p) {
    vec4 leftLeg = sdLeg(p);

    vec3 q = p;
    q.x = -q.x;
    vec4 rightLeg = sdLeg(q);

    return sminVec4(leftLeg, rightLeg, 0.005);
}

vec4 sdArm(vec3 p, bool isLeft) {
    vec3 shoulder = vec3(0.25, 0.5, -0.73);

    vec3 hand;
    float r = 0.07;

    if(isLeft) {
        float t = iTime * 2.0;
        float offset = 0.08 * lazycos(t);
        hand = shoulder + vec3(0.25, offset, -0.02);
    } else {
        hand = shoulder + vec3(0.20, -0.2, -0.02);
    }

    float d = sdCapsule(p, shoulder, hand, r);

    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdArmPair(vec3 p) {
    vec4 L = sdArm(p, true);
    vec3 q = p;
    q.x = -q.x;
    vec4 R = sdArm(q, false);

    return sminVec4(L, R, 0.005);
}

vec4 sdMouth(vec3 p) {
    vec3 mouthPos = vec3(0.0, 0.380, -0.45);
    vec3 lp = p - mouthPos;

    float smileFactor = 0.025;
    lp.y -= smileFactor * (lp.x * lp.x) / (0.07 * 0.07);

    float d = sdTorus(lp, vec2(0.09, 0.035));
    return vec4(d, vec3(0.0, 0.0, 0.02));
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p) {
    float d = 1e10;

    // TODO
    //d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);

    vec3 p2 = p - vec3(0.0, 0.5, -0.7);
    vec3 pa = vec3(0, 0.18, 0);
    vec3 pb = vec3(0, 0, 0);
    float r1 = 0.27;
    float r2 = 0.35;

    d = sdRoundCone(p2, pa, pb, r1, r2);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

const vec3 CAMERA_POS = vec3(0.0, 0.5, 1.0);

vec3 rotateY(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

vec4 sdMonster(vec3 p) {
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);

    float angleY = 3.14159 / 2.0;  // 90° вправо
    //p = rotateY(p, angleY);
    //p +=  vec3(0.6, 0.0, -0.4);

    vec4 legs = sdLegPair(p);
    vec4 body = sdBody(p);
    vec4 eye = sdEye(p);
    vec4 arms = sdArmPair(p);
    vec4 mouth = sdMouth(p);

    float k = 0.005;

    vec4 res = sminVec4(legs, body, k);
    res = sminVec4(res, arms, 0.008);
    res = sminVec4(res, eye, k);
    res = vec4(min(res.x, mouth.x), res.x < mouth.x ? res.yzw : mouth.yzw);

    return res;
}

vec4 sdTotal(vec3 p) {
    vec4 res = sdMonster(p);

    float dist = sdPlane(p);
    if(dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }

    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal(in vec3 p) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps, 0);
    return normalize(vec3(sdTotal(p + h.xyy).x - sdTotal(p - h.xyy).x, sdTotal(p + h.yxy).x - sdTotal(p - h.yxy).x, sdTotal(p + h.yyx).x - sdTotal(p - h.yyx).x));
}

vec4 raycast(vec3 ray_origin, vec3 ray_direction) {

    float EPS = 1e-3;

    // p = ray_origin + t * ray_direction;

    float t = 0.0;

    for(int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t * ray_direction);
        t += res.x;
        if(res.x < EPS) {
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

    if(raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }

    return 1.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.y;

    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5 * wh, -1.0));

    vec4 res = raycast(ray_origin, ray_direction);

    vec3 col = res.yzw;

    vec3 surface_point = ray_origin + res.x * ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = getLightPos();

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));

    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;

    // Output to screen
    fragColor = vec4(col, 1.0);
}