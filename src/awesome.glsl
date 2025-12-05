// =================================================================
// GPGPU EXPERT: Майк Вазовски - Glass & Refraction + Proper Floor
// =================================================================

// --- SDF Primitives ---
float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdPlane(vec3 p, float h) { return p.y - h; }

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
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

mat2 rot(float a) {
    float s = sin(a), c = cos(a);
    return mat2(c, -s, s, c);
}

// --- ТЕКСТУРЫ И ОКРУЖЕНИЕ ---

// Процедурный пол (шахматка) с антиалиасингом (фильтрацией)
vec3 getFloorTexture(vec3 p) {
    // Размер клетки
    float size = 1.0; 
    // Координаты клетки
    vec2 cell = floor(p.xz / size);
    // Шахматный паттерн (0 или 1)
    float checker = mod(cell.x + cell.y, 2.0);
    
    // Цвета пола (темно-серый и светло-серый)
    vec3 col1 = vec3(0.2);
    vec3 col2 = vec3(0.4);
    
    // Затухание шахматки вдаль (чтобы не рябило)
    float fade = smoothstep(5.0, 20.0, length(p.xz));
    vec3 col = mix(col1, col2, checker);
    
    return mix(col, vec3(0.3), fade); // Смешиваем с серым вдалеке
}

vec3 getEnvironment(vec3 rd) {
    // Градиент неба
    vec3 col = mix(vec3(0.6, 0.7, 0.9), vec3(0.1, 0.2, 0.4), rd.y * 0.5 + 0.5);
    // Горизонт
    col = mix(col, vec3(0.7, 0.7, 0.8), pow(1.0 - max(abs(rd.y), 0.0), 8.0));
    return col;
}

// -----------------------------------------------------------------
// ГЕОМЕТРИЯ (SDF)
// -----------------------------------------------------------------

vec4 mapMonster(vec3 p) {
    // --- ANIMATION ---
    float bpm = iTime * 8.0;
    float bounce = abs(sin(bpm)) * 0.05;
    p.y -= bounce; 
    p.xz *= rot(sin(iTime * 3.0) * 0.1);
    p.x += sin(iTime * 2.0) * 0.05;
    p.y -= 0.0; 

    // --- BODY ---
    float squash = 1.0 + 0.05 * sin(bpm);
    vec3 pBody = p - vec3(0.0, 0.4, 0.0);
    pBody.y /= squash; pBody.xz *= sqrt(squash);
    float dBody = sdSphere(pBody, 0.35);
    
    // --- LIMBS ---
    vec3 legL_Start = vec3(-0.15, 0.4, 0.0);
    vec3 legR_Start = vec3( 0.15, 0.4, 0.0);
    vec3 legL_End = vec3(-0.25, bounce * 0.5, 0.1 * sin(iTime)); 
    vec3 legR_End = vec3( 0.25, bounce * 0.5, -0.1 * sin(iTime));
    
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

    // Цвет стекла (поглощение): r, g, b
    vec3 bodyCol = vec3(0.1, 0.9, 0.3); // Зеленое стекло

    // --- HORNS ---
    float hornL = sdSphere(p - vec3(-0.15, 0.72, -0.05), 0.04);
    float hornR = sdSphere(p - vec3( 0.15, 0.72, -0.05), 0.04);
    float dHorns = min(hornL, hornR);

    // Смешивание
    vec2 blendInfo = sminC(dBody, dHorns, 0.05); 
    float dSkin = blendInfo.x;
    float blendFactor = blendInfo.y;
    
    vec3 hornColor = vec3(0.9, 0.8, 0.6); 
    vec3 finalSkinColor = mix(hornColor, bodyCol, blendFactor);

    // --- EYE ---
    vec3 eyeOffset = vec3(sin(iTime)*0.02, cos(iTime*2.0)*0.02, 0.0);
    vec3 eyePos = p - vec3(0.0, 0.5, 0.28) - eyeOffset; 
    
    float dSclera = sdSphere(eyePos, 0.14);
    float dIris = sdSphere(eyePos - vec3(0.0, 0.0, 0.08), 0.07);
    float dPupil = sdSphere(eyePos - vec3(0.0, 0.0, 0.12), 0.035);

    vec3 eyeColor = vec3(0.95);
    float isEye = 0.0; 
    
    if (dIris < dSclera) eyeColor = vec3(0.1, 0.6, 0.8);
    if (dPupil < dIris && dPupil < dSclera) eyeColor = vec3(0.02);

    float dEye = min(dSclera, min(dIris, dPupil));
    float dFinal = smin(dSkin, dEye, 0.01);
    
    vec3 finalColor = finalSkinColor;
    if (dEye < dSkin) {
        finalColor = eyeColor;
        isEye = 1.0; 
    }

    // .w > 0.5 -> Глаз (непрозрачный)
    // .w < 0.5 -> Стекло (прозрачный)
    return vec4(dFinal, finalColor.x, finalColor.y, isEye > 0.5 ? -1.0 : finalColor.z);
}

// Карта только для raymarching'а монстра (без пола)
vec4 map(vec3 p) {
    return mapMonster(p);
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

// Raymarcher (ищет только монстра)
vec4 raycast(vec3 ro, vec3 rd) {
    float t = 0.0;
    for(int i = 0; i < 100; i++) {
        vec3 p = ro + t * rd;
        vec4 res = map(p);
        if(res.x < 0.001) return vec4(t, res.yzw);
        if(t > 20.0) break;
        t += res.x;
    }
    return vec4(-1.0, 0.0, 0.0, 0.0);
}

// Внутренний Raymarcher (для прохода света сквозь стекло)
float raycastInside(vec3 ro, vec3 rd) {
    float t = 0.05; 
    for(int i = 0; i < 40; i++) { 
        vec3 p = ro + t * rd;
        float d = mapMonster(p).x; 
        if(d > 0.0) return t; // Вышли наружу
        t += max(abs(d), 0.02);
        if(t > 5.0) break;
    }
    return t;
}

// Расчет теней (мягкие тени)
// ro - точка на поверхности, rd - направление на свет
float calcShadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0;
    float t = 0.02; // Начинаем чуть дальше от поверхности
    for(int i = 0; i < 32; i++) {
        float h = map(ro + rd * t).x;
        if(h < 0.001) return 0.0; // Полностью в тени
        res = min(res, k * h / t);
        t += h;
        if(t > 10.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Камера
    vec3 ro = vec3(0.0, 0.8, 2.5);
    vec3 lookAt = vec3(0.0, 0.4, 0.0);
    vec3 fwd = normalize(lookAt - ro);
    vec3 right = normalize(cross(vec3(0.0, 1.0, 0.0), fwd));
    vec3 up = cross(fwd, right);
    vec3 rd = normalize(fwd + right * uv.x + up * uv.y);

    // 1. Сначала ищем монстра
    vec4 res = raycast(ro, rd);
    float tMonster = res.x;
    
    // 2. Аналитически ищем пол (y = 0)
    float tPlane = (0.0 - ro.y) / rd.y;
    
    // Определяем, что ближе и что видно
    bool hitMonster = (tMonster > 0.0);
    bool hitPlane = (tPlane > 0.0);
    
    // Если монстр дальше пола, то мы его не видим (если пол перекрывает)
    if (hitMonster && hitPlane && tPlane < tMonster) {
        hitMonster = false;
    }

    vec3 col = vec3(0.0);
    vec3 lightPos = vec3(2.0, 4.0, 3.0);

    // --- РЕНДЕРИНГ ФОНА И ПОЛА ---
    if (hitPlane && !hitMonster) {
        vec3 p = ro + tPlane * rd;
        vec3 n = vec3(0.0, 1.0, 0.0);
        
        // Текстура пола
        vec3 floorCol = getFloorTexture(p);
        
        // Тени на полу от монстра!
        vec3 l = normalize(lightPos - p);
        float shadow = calcShadow(p, l, 8.0);
        
        // Трюк для стеклянной тени:
        // Если тень есть (shadow < 1.0), мы не делаем её черной,
        // а подкрашиваем зеленым светом, прошедшим сквозь монстра.
        vec3 shadowCol = vec3(0.05, 0.3, 0.1); // Цвет каустики
        vec3 lighting = mix(shadowCol, vec3(1.0), shadow);
        
        // Немного ambient
        lighting += 0.2;
        
        col = floorCol * lighting;
        
        // Легкий туман вдаль
        float fog = 1.0 - exp(-0.05 * tPlane);
        col = mix(col, getEnvironment(rd), fog);
    } else {
        col = getEnvironment(rd);
    }

    // --- РЕНДЕРИНГ МОНСТРА ---
    if (hitMonster) {
        vec3 p = ro + tMonster * rd;
        vec3 n = calcNormal(p);
        
        bool isGlass = res.w > -0.5; 
        
        vec3 albedo;
        if (isGlass) albedo = vec3(res.y, res.z, res.w); 
        else albedo = vec3(res.y, res.z, abs(res.w));    
        
        vec3 l = normalize(lightPos - p);
        float diff = max(dot(n, l), 0.0);
        
        if (isGlass) {
            // == СТЕКЛО ==
            vec3 refDir = reflect(rd, n);
            vec3 refCol = getEnvironment(refDir);
            
            float fresnel = 0.05 + 0.95 * pow(1.0 + dot(rd, n), 3.0);
            
            float ior = 1.0 / 1.45;
            vec3 refrDir = refract(rd, n, ior);
            
            vec3 refractCol;
            if(length(refrDir) == 0.0) {
                refractCol = refCol;
            } else {
                float dInside = raycastInside(p, refrDir);
                vec3 pExit = p + refrDir * dInside;
                
                // Проверяем, что видим сквозь стекло: пол или небо?
                float tFloorInternal = (0.0 - p.y) / refrDir.y;
                
                // Если луч преломления идет вниз и попадает в пол
                if (tFloorInternal > 0.0 && tFloorInternal < dInside + 10.0) {
                     vec3 pFloor = p + refrDir * tFloorInternal;
                     refractCol = getFloorTexture(pFloor);
                     // Затенение пола под стеклом
                     refractCol *= 0.8; 
                } else {
                     refractCol = getEnvironment(refrDir);
                }
                
                // Поглощение (зеленая толща)
                vec3 absorb = exp(-albedo * dInside * 2.5);
                refractCol *= absorb;
            }
            
            col = mix(refractCol, refCol, fresnel);
            
            // Спекуляр
            vec3 h = normalize(l - rd);
            float spec = pow(max(dot(n, h), 0.0), 64.0);
            col += vec3(1.0) * spec * 0.8;
            
        } else {
            // == ГЛАЗ (Пластик) ==
            // Тень на глазу от века (самозатенение)
            float selfShadow = calcShadow(p + n*0.01, l, 16.0);
            
            vec3 h = normalize(l - rd);
            float spec = pow(max(dot(n, h), 0.0), 32.0);
            vec3 amb = vec3(0.1);
            col = albedo * (diff * selfShadow + amb) + vec3(1.0)*spec*selfShadow;
        }
    }

    col = pow(col, vec3(0.4545));
    fragColor = vec4(col, 1.0);
}
