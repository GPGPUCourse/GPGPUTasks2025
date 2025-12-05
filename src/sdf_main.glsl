// =================================================================
// GPGPU EXPERT: Полный код для рендеринга Майка Вазовски (SDF)
// =================================================================

// Вспомогательные функции (SDF Primitives)
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdPlane(vec3 p, float h) {
    return p.y - h;
}

vec2 sminC(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    float d = mix(b, a, h) - k * h * (1.0 - h);
    return vec2(d, h);
}

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Матрица вращения 2D
mat2 rot(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

// -----------------------------------------------------------------
// ГЕОМЕТРИЯ ПЕРСОНАЖА
// -----------------------------------------------------------------

vec4 mapMonster(vec3 p) {
    float bpm = iTime * 8.0; // Скорость бита
    
    // 1. Прыжок (Bounce)
    float bounce = abs(sin(bpm)) * 0.05;
    p.y -= bounce; 
    
    // 2. Покачивание бедрами (Twist)
    p.xz *= rot(sin(iTime * 3.0) * 0.1);
    
    // 3. Наклон тела (Sway)
    p.x += sin(iTime * 2.0) * 0.05;

    p.y -= 0.0; 

    // --- 1. ТЕЛО (Зеленое) ---
    float squash = 1.0 + 0.05 * sin(bpm);
    vec3 pBody = p - vec3(0.0, 0.4, 0.0);
    pBody.y /= squash;
    pBody.xz *= sqrt(squash);
    
    float dBody = sdSphere(pBody, 0.35);
    
    // --- КОНЕЧНОСТИ (Анимированные) ---
    
    vec3 legL_Start = vec3(-0.15, 0.4, 0.0);
    vec3 legR_Start = vec3( 0.15, 0.4, 0.0);
    vec3 legL_End = vec3(-0.25, 0.0 + bounce * 0.5, 0.1 * sin(iTime)); 
    vec3 legR_End = vec3( 0.25, 0.0 + bounce * 0.5, -0.1 * sin(iTime));
    
    float legL = sdCapsule(p, legL_Start, legL_End, 0.06);
    float legR = sdCapsule(p, legR_Start, legR_End, 0.06);
    
    float armWave = sin(iTime * 5.0) * 0.2;
    float armFlap = abs(cos(iTime * 5.0)) * 0.3;
    
    vec3 armL_End = vec3(-0.6, 0.5 + armFlap, 0.2 + armWave);
    vec3 armR_End = vec3( 0.6, 0.5 + armFlap + 0.1, 0.2 - armWave);

    float armL = sdCapsule(p, vec3(-0.3, 0.5, 0.0), armL_End, 0.05);
    float armR = sdCapsule(p, vec3( 0.3, 0.5, 0.0), armR_End, 0.05);

    dBody = smin(dBody, legL, 0.1);
    dBody = smin(dBody, legR, 0.1);
    dBody = smin(dBody, armL, 0.08);
    dBody = smin(dBody, armR, 0.08);

    vec3 bodyCol = vec3(0.1, 0.8, 0.1);

    // --- 2. РОГА (Бежевые) ---
    float hornL = sdSphere(p - vec3(-0.15, 0.72, -0.05), 0.04);
    float hornR = sdSphere(p - vec3( 0.15, 0.72, -0.05), 0.04);
    float dHorns = min(hornL, hornR);

    // --- 3. СМЕШИВАНИЕ ТЕЛА И РОГОВ ---
    vec2 blendInfo = sminC(dBody, dHorns, 0.05); 
    float dSkin = blendInfo.x;
    float blendFactor = blendInfo.y;

    vec3 hornColor = vec3(0.9, 0.85, 0.7);
    vec3 finalSkinColor = mix(hornColor, bodyCol, blendFactor);

    // --- 4. ГЛАЗ (Сложная структура) ---
    vec3 eyeOffset = vec3(sin(iTime)*0.02, cos(iTime*2.0)*0.02, 0.0);
    vec3 eyePos = p - vec3(0.0, 0.5, 0.28) - eyeOffset; 
    
    float dSclera = sdSphere(eyePos, 0.14);
    float dIris = sdSphere(eyePos - vec3(0.0, 0.0, 0.08), 0.07);
    float dPupil = sdSphere(eyePos - vec3(0.0, 0.0, 0.12), 0.035);

    vec3 eyeColor = vec3(0.95);
    if (dIris < dSclera) eyeColor = vec3(0.1, 0.6, 0.8);
    if (dPupil < dIris && dPupil < dSclera) eyeColor = vec3(0.05);

    float dEye = min(dSclera, min(dIris, dPupil));

    float dFinal = smin(dSkin, dEye, 0.01);
    
    vec3 finalColor = finalSkinColor;
    if (dEye < dSkin) {
        finalColor = eyeColor;
    }

    return vec4(dFinal, finalColor);
}

vec4 map(vec3 p) {
    vec4 res = mapMonster(p);

    float dPlane = sdPlane(p, 0.0);
    
    if (dPlane < res.x) {
        res = vec4(dPlane, vec3(0.2 + 0.1*mod(floor(p.x)+floor(p.z), 2.0))); 
    }

    return res;
}

vec3 calcNormal(vec3 p) {
    const float eps = 0.0005;
    const vec2 h = vec2(eps, 0);
    return normalize(vec3(
        map(p + h.xyy).x - map(p - h.xyy).x,
        map(p + h.yxy).x - map(p - h.yxy).x,
        map(p + h.yyx).x - map(p - h.yyx).x
    ));
}

// Raymarching (Трассировка лучей)
vec4 raycast(vec3 ro, vec3 rd) {
    float t = 0.0;
    vec3 col = vec3(0.0);
    
    for(int i = 0; i < 100; i++) {
        vec3 p = ro + t * rd;
        vec4 res = map(p);
        
        if(res.x < 0.001) {
            return vec4(t, res.yzw);
        }
        if(t > 20.0) break;
        
        t += res.x;
        col = res.yzw;
    }
    
    return vec4(-1.0, 0.0, 0.0, 0.0);
}

float calcShadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0;
    float t = 0.01;
    for(int i = 0; i < 32; i++) {
        float h = map(ro + rd * t).x;
        if(h < 0.001) return 0.0;
        res = min(res, k * h / t);
        t += h;
        if(t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    vec3 ro = vec3(0.0, 0.8, 2.5);
    vec3 lookAt = vec3(0.0, 0.4, 0.0);
    
    vec3 fwd = normalize(lookAt - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    
    vec3 rd = normalize(fwd + right * uv.x + up * uv.y);

    vec4 res = raycast(ro, rd);
    float t = res.x;
    vec3 col = vec3(0.05);

    if(t > 0.0) {
        vec3 p = ro + t * rd;
        vec3 n = calcNormal(p);
        vec3 albedo = res.yzw;
        
        vec3 lightPos = vec3(2.0, 4.0, 3.0);
        vec3 l = normalize(lightPos - p);
        
        float diff = max(dot(n, l), 0.0);
        
        float shadow = calcShadow(p + n * 0.01, l, 16.0);
        
        vec3 viewDir = normalize(ro - p);
        vec3 reflectDir = reflect(-l, n);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        
        float amb = 0.2 + 0.5 * n.y;
        
        vec3 lin = vec3(0.0);
        lin += diff * vec3(1.0) * shadow;
        lin += amb * vec3(0.4);
        
        col = albedo * lin;
        col += spec * 0.3 * shadow;
        
        col = pow(col, vec3(0.4545));
    }

    fragColor = vec4(col, 1.0);
}
