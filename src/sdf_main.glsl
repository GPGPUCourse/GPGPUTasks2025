const int ITERS_COUNT = 256;
const float EPS = 0.001;
const float MAX_Y = 2.0;

const float PI = 3.1415926535;

float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }

float smin(float a, float b, float k)
{
    float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0);
    return mix(b, a, h) - k*h*(1.0-h);
}

float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

float sdRoundBox(vec3 p, vec3 b, float r)
{
  vec3 q = abs(p) - b + r;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float sdPlane(vec3 p)
{
    return p.y;
}

float sdEllipsoid(vec3 p, vec3 r)
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
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

  float k = sign(rr)*rr*rr*x2;
  if( sign(z)*a2*z2>k ) return  sqrt(x2 + z2)        *il2 - r2;
  if( sign(y)*a2*y2<k ) return  sqrt(x2 + y2)        *il2 - r1;
                        return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

float sdRoundedCylinderHorizontal(vec3 p, float ra, float rb, float h) {
    vec2 d = vec2(length(p.yz)-ra+rb, abs(p.x) - h + rb);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

vec3 barbellPos() {
    vec3 center = vec3(0.0, 1.0, -0.3);
    float limit = 0.3;
    float len = 0.3;
    return center + vec3(0.0,
        clamp(sin(iTime * 2.0), -sqrt(1.0 - limit * limit), sqrt(1.0 - limit * limit)),
        max(abs(cos(iTime * 2.0)), limit)) * len;
}

vec4 sdBarbell(vec3 p) {
    float d = 1e10;
    float weightH = 0.025;
    float weightR = 0.3;
    float weightCount = 3.0;
    float s = weightH * 2.0 + 0.005;
    vec3 color = vec3(0.2, 0.2, 0.2);
    
    vec3 pq = p;
    pq.x = abs(pq.x);
    vec3 q = pq - vec3(0.75, 0.0, 0.0);
    q.x = q.x - s*clamp(round(q.x/s), 0.0, 6.0);
    d = sdRoundedCylinderHorizontal(q, weightR, 0.001, weightH);
    
    float barbell = sdRoundedCylinderHorizontal(p, 0.03, 0.001, 1.1);
    barbell = min(barbell, sdRoundedCylinderHorizontal(pq - vec3(0.7, 0.0, 0.0), 0.05, 0.001, 0.05));
    
    if (barbell < d) {
        d = barbell;
        color = vec3(0.7, 0.7, 0.7);
    }
    
    return vec4(d, color);
}

vec4 sdBody(vec3 p)
{
    float d = 1e10;
    d = sdSphere((p - vec3(0.0, 1.4, -0.7)), 0.4);
    d = smin(d, sdSphere((p - vec3(0.0, 1.7, -0.7)), 0.3), 0.25);
    
    vec3 q = p;
    q.x = abs(q.x);
    
    // Chest
    d = smin(d, sdRoundBox((p - vec3(0.17, 1.45, -0.35)), vec3(0.12), 0.07), 0.15);
    d = smin(d, sdRoundBox((p - vec3(-0.17, 1.45, -0.35)), vec3(0.12), 0.07), 0.15);
    
    // Abs
    d = smin(d, sdRoundBox(p - vec3(0.085, 1.3, -0.33), vec3(0.07), 0.03), 0.07);
    d = smin(d, sdRoundBox(p - vec3(-0.085, 1.3, -0.33), vec3(0.07), 0.03), 0.07);
    d = smin(d, sdRoundBox(p - vec3(0.085, 1.15, -0.35), vec3(0.07), 0.03), 0.07);
    d = smin(d, sdRoundBox(p - vec3(-0.085, 1.15, -0.35), vec3(0.07), 0.03), 0.07);
    
    // Legs
    d = smin(d, sdRoundCone(q, vec3(0.175, 1.1, -0.7), vec3(0.2, 0.6, -0.6), 0.15, 0.12), 0.07);
    d = smin(d, sdRoundCone(q, vec3(0.2, 0.6, -0.6), vec3(0.18, 0.1, -0.7), 0.12, 0.1), 0.07);
    d = smin(d, sdRoundBox(q - vec3(0.18, 0.03, -0.5), vec3(0.1, 0.05, 0.2), 0.03), 0.07);
    
    // Arms
    d = smin(d, sdEllipsoid(q - vec3(0.4, 1.3, -0.5), vec3(0.12, 0.3, 0.1)), 0.07);
    vec3 bPos = barbellPos() + vec3(0.4, 0.0, 0.0);
    d = smin(d, sdRoundCone(q, vec3(0.43, 1.0, -0.5), bPos - normalize(bPos - vec3(0.43, 1.0, -0.5)) * 0.12, 0.1, 0.08), 0.07);
    d = smin(d, sdSphere(q - bPos, 0.1), 0.03);
    
    return vec4(d, vec3(0.1, 1.0, 0.1));
}

vec4 sdEye(vec3 p)
{
    vec3 eyePos = vec3(0.0, 1.73, -0.4);
    float d = sdSphere((p - eyePos), 0.15);
    vec3 eyeDir = normalize(vec3(0.0, 0.3, 1.0));
    vec3 color = vec3(1.0, 1.0, 1.0);
    vec3 pointDir = normalize(p - eyePos);
    if (dot(eyeDir, pointDir) >= 0.95) {
        color = vec3(0.0, 0.0, 0.0);
    }
    else if (dot(eyeDir, pointDir) >= 0.8) {
        color = vec3(0.0, 1.0, 1.0);
    }
    return vec4(d, color);
}

vec4 sdHorseCocker(vec3 p)
{
    vec4 res = sdBody(p);

    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }

    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = vec4(1e10, 0.0, 0.0, 0.0);
    float s = 4.0;
    {
        vec3 q = p - vec3(0.0, 0.0, -1.0);
        q.x = q.x - s*round(q.x/s);
        q.z = q.z - s*round(q.z/s);
        res = sdHorseCocker(q);
    }
    
    {
        vec3 q = p - vec3(0.0, 0.0, -1.0) - barbellPos();
        q.x = q.x - s*round(q.x/s);
        q.z = q.z - s*round(q.z/s);
        vec4 barbell = sdBarbell(q);
        if (barbell.x < res.x) {
            res = barbell;
        }
    }

    float dist = sdPlane(p);
    if (dist < res.x) {
        float step = 1.0;
        if (bool(int(p.x - step * floor(p.x / step) > step / 2.0) ^
            int(p.z - step * floor(p.z / step) > step / 2.0))) {
            res = vec4(dist, vec3(1.0));
        } else {
            res = vec4(dist, vec3(0.0));
        }
    }

    return res;
}

vec3 calcNormal(in vec3 p)
{
    const float h = 0.0001;
    #define ZERO (min(iFrame,0))
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*sdTotal(p+e*h).x;
    }
    return normalize(n);
    /*const float eps = 0.0001;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
    sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
    sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );*/
}


struct RayCastResult
{
    float t;
    vec3 color;
    bool hasHit;
};

RayCastResult raycast(vec3 rayOrig, vec3 rayDir)
{
    float t = 0.0;
    for (int iter = 0; iter < ITERS_COUNT; ++iter) {
        vec4 res = sdTotal(rayOrig + t * rayDir);
        t += res.x;
        if (res.x < EPS) {
            return RayCastResult(t, res.yzw, true);
        }
        
        if (rayOrig.y + rayDir.y * t > MAX_Y){
            break;
        }
    }
    return RayCastResult(t, vec3(0.0, 0.0, 0.0), false);
}

vec3 shade(vec3 normal, vec3 ka, vec3 kd, vec3 ks, float shininess, vec3 lightDir, vec3 lightColor, float shadow, vec3 cameraDir)
{
    vec3 L = -lightDir;
    vec3 V = -cameraDir;
    vec3 H = normalize(L + V);
    
    vec3 ambient = ka * lightColor;
    
    float diff = max(dot(normal, L), 0.0) * shadow;
    vec3 diffuse = kd * diff * lightColor;
    
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(normal, H), 0.0), shininess);
    }
    vec3 specular = ks * spec * lightColor;
    
    return ambient + diffuse + specular;
}

float castShadowDirLight(vec3 p, vec3 lightDir)
{
    float w = 0.05;
    float res = 1.0;
    float ph = 1e20;
    float t = 0.005;
    for(int i = 0; i < ITERS_COUNT; ++i)
    {
        float h = sdTotal(p - lightDir * t).x;
        if(h < 0.001) {
            return 0.0;
        }
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        res = min(res, d / (w * max(0.0, t - y)));
        ph = h;
        t += h;
        if (p.y - lightDir.y * t > MAX_Y){
            break;
        }
    }
    return res;
}

struct DirLight {
    vec3 dir;
    vec3 color;
};

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);
    
    vec3 cameraPos = vec3(0.0, 1.8, 1.0);
    vec3 cameraDir = vec3(0.0, 0.0, -1.0);
    
    vec3 rayOrig = cameraPos;
    vec3 rayDir = normalize(vec3(uv - 0.5*wh, -1.0));

    RayCastResult res = raycast(rayOrig, rayDir);
    
    const int dirLightsCount = 4;
    float colorIntens = 0.8;
    DirLight dirLights[dirLightsCount] = DirLight[dirLightsCount](
        DirLight(normalize(vec3(1.0, -1.0, -1.0)), vec3(1.0) * 0.3),
        DirLight(normalize(vec3(sin(iTime * PI), -0.5, cos(iTime * PI))), vec3(1.0, 0.0, 0.0) * colorIntens),
        DirLight(normalize(vec3(sin(iTime * PI + PI * 2.0 / 3.0), -0.5, cos(iTime * PI + PI * 2.0 / 3.0))), vec3(0.0, 1.0, 0.0) * colorIntens),
        DirLight(normalize(vec3(sin(iTime * PI + PI * 4.0 / 3.0), -0.5, cos(iTime * PI + PI * 4.0 / 3.0))), vec3(0.0, 0.0, 1.0) * colorIntens)
    );
    
    if (!res.hasHit) {
        fragColor = mix(vec4(0.3, 0.3, 0.9, 1.0), vec4(0.1, 0.1, 0.3, 1.0), (uv.y - 0.5) * 2.0);
        return;
    }
    
    /*vec3 ka = res.color * 0.1;
    vec3 kd = res.color * 0.9;
    vec3 ks = vec3(0.1);
    float shininess = 30.0;*/
    
    vec3 ka = res.color * 0.1;
    vec3 kd = res.color * 0.8;
    vec3 ks = vec3(0.2);
    float shininess = 30.0;
    
    
    /*vec3 ka = res.color * 0.1;
    vec3 kd = res.color * 0.6;
    vec3 ks = vec3(0.3);
    float shininess = 30.0;*/

    vec3 pos = rayOrig + res.t * rayDir;
    vec3 normal = calcNormal(pos);
    
    vec3 col = ka * 0.05;
    
    for (int i = 0; i < dirLightsCount; ++i) {
        DirLight light = dirLights[i];
        float shadow = castShadowDirLight(pos, light.dir);
        col += shade(normal, ka, kd, ks, shininess, light.dir, light.color, shadow, cameraDir);
    }

    fragColor = vec4(col, 1.0);
}
