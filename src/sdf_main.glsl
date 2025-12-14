float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdPlane(vec3 p)
{
    return p.y;
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

vec3 monsterSpace(vec3 p)
{
    vec3 center = vec3(0.0, 0.35, -0.7);
    return p - center;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

float sdCappedCylinder(vec3 p, float h, float r)
{
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

vec3 rotX(vec3 p, float a)
{
    float c = cos(a);
    float s = sin(a);
    p.yz = mat2(c, -s, s, c) * p.yz;
    return p;
}

float sdPinShape(vec3 p)
{
    float body = sdCapsule(p, vec3(0.0, 0.00, 0.0), vec3(0.0, 0.22, 0.0), 0.06);
    float head = sdSphere(p - vec3(0.0, 0.26, 0.0), 0.055);
    return min(body, head);
}

vec4 sdPins(vec3 p)
{
    float d = 1e10;
    vec3 col = vec3(1.0);

    float t = mod(iTime, 4.0);
    float hit = smoothstep(3.1, 3.6, t);

    vec3 base = vec3(0.0, 0.0, -3.6);

    int idx = 0;
    for (int row = 0; row < 4; ++row) {
        for (int j = 0; j <= row; ++j) {
            float fx = (float(j) - 0.5 * float(row)) * 0.18;
            float fz = float(row) * 0.20;

            vec3 pinPos = base + vec3(fx, 0.0, -fz);

            float dir = (float(idx) * 1.7 + float(row) * 2.3);
            float side = sin(dir);

            float a = hit * (1.2 + 0.3 * sin(dir));

            vec3 q = p - pinPos;
            q.y -= 0.13;
            q.x += 0.08 * side;
            q = rotX(q, a);
            q.y += 0.13;

            float di = sdPinShape(q);
            if (di < d) {
                d = di;
                col = vec3(0.97, 0.97, 0.98);
            }

            idx++;
        }
    }

    return vec4(d, col);
}

vec4 sdBody(vec3 p)
{
    vec3 q = monsterSpace(p);
    float d = sdSphere(q, 0.35);
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec3 q = monsterSpace(p);

    vec3 eyeL = vec3(-0.12, 0.08, 0.25);
    vec3 eyeR = vec3( 0.12, 0.08, 0.25);

    float d = 1e10;
    vec3 col = vec3(0.0);

    float dEyeL = sdSphere(q - eyeL, 0.08);
    float dEyeR = sdSphere(q - eyeR, 0.08);

    d = dEyeL;
    col = vec3(1.0);
    if (dEyeR < d) {
        d = dEyeR;
        col = vec3(1.0);
    }

    vec3 irisColorOuter = vec3(0.35, 0.80, 1.00);
    vec3 irisColorInner = vec3(0.05, 0.25, 0.70);


    vec3 irisOff = vec3(0.0, 0.0, 0.02);

    float dIrisL = sdSphere(q - (eyeL + irisOff), 0.055);
    if (dIrisL < d) {
        float k = clamp(length(q - (eyeL + irisOff)) / 0.055, 0.0, 1.0);
        col = mix(irisColorInner, irisColorOuter, k);
        d = dIrisL;
    }

    float dIrisR = sdSphere(q - (eyeR + irisOff), 0.055);
    if (dIrisR < d) {
        float k = clamp(length(q - (eyeR + irisOff)) / 0.055, 0.0, 1.0);
        col = mix(irisColorInner, irisColorOuter, k);
        d = dIrisR;
    }

    float dPupilL = sdSphere(q - (eyeL + vec3(0.0, 0.0, 0.03)), 0.028);
    if (dPupilL < d) {
        d = dPupilL;
        col = vec3(0.0);
    }

    float dPupilR = sdSphere(q - (eyeR + vec3(0.0, 0.0, 0.03)), 0.028);
    if (dPupilR < d) {
        d = dPupilR;
        col = vec3(0.0);
    }

    float dGlintL = sdSphere(q - (eyeL + vec3(-0.02, 0.02, 0.045)), 0.015);
    if (dGlintL < d) {
        d = dGlintL;
        col = vec3(1.0);
    }

    float dGlintR = sdSphere(q - (eyeR + vec3(-0.02, 0.02, 0.045)), 0.015);
    if (dGlintR < d) {
        d = dGlintR;
        col = vec3(1.0);
    }

    return vec4(d, col);
}

vec4 sdHair(vec3 p)
{
    vec3 q = monsterSpace(p);

    float d = 1e10;
    vec3 col = vec3(0.1, 0.2, 1.0);

    for (int i = 0; i < 6; ++i) {
        float z = -0.08 + 0.04 * float(i);
        vec3 hPos = vec3(0.0, 0.32, z);
        float di = sdSphere(q - hPos, 0.05);
        if (di < d) d = di;
    }

    return vec4(d, col);
}

vec4 sdHat(vec3 p)
{
    vec3 q = monsterSpace(p);

    vec3 c = vec3(0.0, 0.33, 0.02);

    float brim = sdCappedCylinder(q - c, 0.015, 0.18);

    vec3 topP = q - (c + vec3(0.0, 0.07, 0.0));
    float top = sdCappedCylinder(topP, 0.07, 0.11);

    float d = min(brim, top);
    vec3 col = vec3(0.08, 0.12, 0.35);

    return vec4(d, col);
}

vec4 sdTeeth(vec3 p)
{
    vec3 q = monsterSpace(p);
    float d = 1e10;
    vec3 col = vec3(1.0);

    vec3 t0 = vec3(0.0,-0.02, 0.28);
    float d0 = sdSphere(q - t0, 0.03);
    d = d0;
    col = vec3(1.0);

    vec3 tL = vec3(-0.05,-0.03, 0.27);
    float dL = sdSphere(q - tL, 0.025);
    if (dL < d) {
        d = dL;
        col = vec3(1.0);
    }

    vec3 tR = vec3( 0.05,-0.03, 0.27);
    float dR = sdSphere(q - tR, 0.025);
    if (dR < d) {
        d = dR;
        col = vec3(1.0);
    }

    vec3 mA = vec3(-0.06,-0.06, 0.24);
    vec3 mB = vec3( 0.06,-0.06, 0.24);
    float dM = sdCapsule(q, mA, mB, 0.02);
    if (dM < d) {
        d = dM;
        col = vec3(0.1, 0.0, 0.0);
    }

    vec3 b0 = vec3(0.0,-0.11,0.32);
    float dB0 = sdSphere(q - b0, 0.04);
    if (dB0 < d) {
        d = dB0;
        col = vec3(0.25, 0.15, 0.05);
    }

    vec3 bL = vec3(-0.035,-0.115,0.33);
    float dBL = sdSphere(q - bL, 0.035);
    if (dBL < d) {
        d = dBL;
        col = vec3(0.25, 0.15, 0.05);
    }

    vec3 bR = vec3(0.035,-0.115,0.33);
    float dBR = sdSphere(q - bR, 0.035);
    if (dBR < d) {
        d = dBR;
        col = vec3(0.25, 0.15, 0.05);
    }

    return vec4(d, col);
}

vec4 sdLimbs(vec3 p)
{
    vec3 q = monsterSpace(p);

    float d = 1e10;
    vec3 col = vec3(0.0, 0.9, 0.0);

    float phase = 0.5 - 0.5 * cos(iTime);
    float spread = mix(0.0, 0.35, phase);

    float t = mod(iTime, 4.0);
    float throwPhase = smoothstep(1.5, 2.2, t);

    float wave = lazycos(iTime * 2.0);
    float armAngle = 0.6 * wave;
    armAngle = mix(armAngle, -1.1, throwPhase);

    float c = cos(armAngle);
    float s = sin(armAngle);

    vec3 aL = vec3(-0.25, 0.05, 0.0);
    vec3 bL = vec3(-0.45,-0.15, 0.0);
    float dArmL = sdCapsule(q, aL, bL, 0.06);
    if (dArmL < d) d = dArmL;

    vec3 shoulderR  = vec3(0.25, 0.05, 0.0);
    vec3 handR_rest = vec3(0.45,-0.15, 0.0);
    vec3 rel = handR_rest - shoulderR;
    rel.yz = mat2(c, -s, s, c) * rel.yz;
    vec3 handR = shoulderR + rel;

    float dArmR = sdCapsule(q, shoulderR, handR, 0.06);
    if (dArmR < d) d = dArmR;

    vec3 lLa = vec3(-0.12,-0.25,-0.05);
    vec3 lLb = vec3(-0.12,-0.36,-0.05);
    lLb.x -= spread;
    float dLegL = sdCapsule(q, lLa, lLb, 0.07);
    if (dLegL < d) d = dLegL;

    vec3 lRa = vec3( 0.12,-0.25,-0.05);
    vec3 lRb = vec3( 0.12,-0.36,-0.05);
    lRb.x += spread;
    float dLegR = sdCapsule(q, lRa, lRb, 0.07);
    if (dLegR < d) d = dLegR;


    if (t < 2.0) {
        vec3 ballPos = handR;
        ballPos.y -= 0.05;
        float dBall = sdSphere(q - ballPos, 0.08);
        if (dBall < d) {
            d = dBall;
            col = vec3(0.05, 0.05, 0.05);
        }
    }

    return vec4(d, col);
}


vec4 sdBowlingBall(vec3 p)
{
    float t = mod(iTime, 4.0);

    if (t < 2.0)
        return vec4(1e10, 0.0, 0.0, 0.0);

    float fly = smoothstep(2.0, 3.2, t);
    float boom = smoothstep(3.2, 4.0, t);

    vec3 start = vec3(0.0, 0.12, -3.5);
    vec3 end   = vec3(0.0, 0.35, 0.8);
    vec3 center = mix(start, end, fly);

    float r = mix(0.12, 6.0, boom);

    float d = sdSphere(p - center, r);
    vec3 col = vec3(0.02);

    return vec4(d, col);
}

vec4 sdExplosion(vec3 p)
{
    float t = iTime;

    float d = 1e10;
    vec3 col = vec3(0.2, 0.8, 1.0);

    for (int i = 0; i < 3; ++i) {
        float fi = float(i);
        float ph = t * 1.5 + fi * 0.8;
        float slide = 0.5 * sin(ph);
        vec3 center = vec3(slide, 0.35, -1.4 - 0.4 * fi);
        float di = sdSphere(p - center, 0.28);
        if (di < d) d = di;
    }

    return vec4(d, col);
}

vec4 sdMonster(vec3 p)
{
    p -= vec3(0.0, 0.08, 0.0);

    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) res = eye;

    vec4 limbs = sdLimbs(p);
    if (limbs.x < res.x) res = limbs;

    vec4 teeth = sdTeeth(p);
    if (teeth.x < res.x) res = teeth;

    vec4 hair = sdHair(p);
    if (hair.x < res.x) res = hair;

    vec4 hat = sdHat(p);
    if (hat.x < res.x) res = hat;

    return res;
}

vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);

    vec4 ball = sdBowlingBall(p);
    if (ball.x < res.x) res = ball;

    vec4 pins = sdPins(p);
    if (pins.x < res.x) res = pins;

    vec4 boom = sdExplosion(p);
    if (boom.x < res.x) res = boom;

    float dist = sdPlane(p);
    if (dist < res.x) {

        float laneW = 0.65;
        float x = abs(p.x);

        float plank = 0.5 + 0.5 * sin(30.0 * p.z + 6.0 * sin(3.0 * p.z));
        vec3 woodA = vec3(0.62, 0.42, 0.20);
        vec3 woodB = vec3(0.48, 0.32, 0.15);
        vec3 laneCol = mix(woodA, woodB, plank);

        float gutter = smoothstep(laneW, laneW + 0.06, x);
        vec3 gutterCol = vec3(0.06, 0.06, 0.07);

        vec3 col = mix(laneCol, gutterCol, gutter);
        res = vec4(dist, col);
    }

    return res;
}


vec3 calcNormal(in vec3 p)
{
    const float eps = 0.001;
    const vec2 h = vec2(eps, 0);
    return normalize(vec3(
        sdTotal(p + h.xyy).x - sdTotal(p - h.xyy).x,
        sdTotal(p + h.yxy).x - sdTotal(p - h.yxy).x,
        sdTotal(p + h.yyx).x - sdTotal(p - h.yyx).x
    ));
}

vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    float EPS = 1e-3;
    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t * ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
        if (t > 50.0) break;
    }

    return vec4(1e10, vec3(0.0));
}

float shading(vec3 p, vec3 light_source, vec3 normal)
{
    vec3 light_dir = normalize(light_source - p);
    float sh = dot(light_dir, normal);
    return clamp(sh, 0.5, 1.0);
}

float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);
    vec3 V = normalize(camera_center - p);
    return pow(max(dot(R, V), 0.0), shinyness);
}

float castShadow(vec3 p, vec3 light_source)
{
    vec3 toP = p - light_source;
    float target_dist = length(toP);
    vec3 dir = normalize(toP);

    float bias = 0.03;
    float hit = raycast(light_source + dir * bias, dir).x;

    if (hit + 0.001 < target_dist) return 0.5;
    return 1.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.y;
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5 * wh, -1.0));

    vec4 res = raycast(ray_origin, ray_direction);

    if (res.x > 1e9) {
        vec3 top = vec3(0.02, 0.03, 0.06);
        vec3 bot = vec3(0.12, 0.10, 0.08);
        float k = clamp(uv.y, 0.0, 1.0);
        vec3 bg = mix(bot, top, k);
        fragColor = vec4(bg, 1.0);
        return;
    }

    vec3 col = res.yzw;

    vec3 surface_point = ray_origin + res.x * ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5 * sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0) * spec;


    float t = mod(iTime, 4.0);
    float fade = smoothstep(3.4, 3.9, t);
    col = mix(col, vec3(0.0), fade * 0.6);

    col = col / (col + vec3(1.0));
    col = pow(col, vec3(1.0 / 2.2));

    fragColor = vec4(col, 1.0);
}
