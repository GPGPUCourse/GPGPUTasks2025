
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

float lazysin(float angle)
{
    return lazycos(angle - 1.57079632679);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r ) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float sdRoundCone( vec3 p, float r1, float r2, float h ) {
    vec2 q = vec2( length(p.xz), p.y );
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
    return dot(q, vec2(a,b)) - r1;
}

float opSmoothSub( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h);
}

float smin( float a, float b, float k ) {
    k *= 4.0;
    float h = max( k-abs(a-b), 0.0 )/k;
    return min(a,b) - h*h*k*(1.0/4.0);
}

vec4 opUnion(vec4 d1, vec4 d2) {
    return (d1.x < d2.x) ? d1 : d2;
}

vec4 sdBody(vec3 p) {
    float top = sdSphere((p - vec3(0.0, 0.76, -0.7)), 0.19);
    float bottom = sdSphere((p - vec3(0.0, 0.4, -0.7)), 0.365);
    return vec4(smin(top, bottom, 0.08), vec3(0.0, 1.0, 0.0));
}

vec4 sdEye(vec3 p)
{
    vec4 dEye = vec4(sdSphere(p - vec3(0, 0.6, -0.4), 0.17), 255, 255, 255);
    vec4 dIris = vec4(sdSphere(p - vec3(0, 0.59, -0.25), 0.1), 0, 255, 255);
    vec4 dPupil = vec4(sdSphere(p - vec3(0, 0.58, -0.2), 0.066), 0, 0, 0);
    float min_x = min(dEye.x, min(dIris.x, dPupil.x));
    if (dIris.x == min_x) {
        return dIris;
    }
    if (dPupil.x == min_x) {
        return dPupil;
    }
    return dEye;
}

float sdVerticalCapsule( vec3 p, float h, float r )
{
  p.y -= clamp( p.y, 0.0, h );
  return length( p ) - r;
}

vec3 rotateAroundZ(vec3 vectorToRotate, float angle) 
{
    float cosAngle = lazycos(angle);
    float sinAngle = lazysin(angle);

    vec2 rotatedXY = mat2(cosAngle, -sinAngle, sinAngle, cosAngle) * vectorToRotate.xy;
    return vec3(rotatedXY, vectorToRotate.z);
}

vec4 sdArms(vec3 p) 
{

    float dArmR = sdCapsule(p, vec3( 0.27, 0.55, -0.7), vec3( 0.41, 0.35, -0.7), 0.05);

    vec3 leftArm_A = vec3(-0.27, 0.55, -0.7);
    vec3 leftArm_B_rest = vec3(-0.41, 0.35, -0.7);

    vec3 leftArmVector = leftArm_B_rest - leftArm_A;
    
    float swingAngle = -0.85 * (lazycos(10.0 * iTime) - 1.0);

    vec3 rotatedArm = rotateAroundZ(leftArmVector, swingAngle);
    vec3 leftArmWaving = leftArm_A + rotatedArm;

    float dArmL = sdCapsule(p, leftArm_A, leftArmWaving, 0.05);

    return vec4(min(dArmR, dArmL), vec3(0.0, 1.0, 0.0));
}

vec4 sdLegs(vec3 p) {

    float legR = sdVerticalCapsule(p - vec3(0.095, -0.01, -0.7), 0.1, 0.058);
    float legL = sdVerticalCapsule(p - vec3(-0.095, -0.01, -0.7), 0.1, 0.058);

    return vec4(min(legR, legL), vec3(0.0, 1.0, 0.0));
}

vec4 sdMonster(vec3 p) {
    p -= vec3(0.5, 0.08, 0.0);
    vec4 res = sdBody(p);
    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }
    vec4 arms = sdArms(p);
    if (arms.x < res.x) {
        res = arms;
    }

    vec4 legs = sdLegs(p);
    if (legs.x < res.x) {
        res = legs;
    }
    return res; 
}


vec4 sdOmNom(vec3 p) {
    p -= vec3(-0.75, 0.417, 0.0); 
    
    vec3 pTeeth = p - vec3(0.0, -0.1, 0.33); 
    pTeeth.x = abs(pTeeth.x); 
    float tooth1 = sdRoundCone(pTeeth - vec3(0.035, 0.0, 0.0), 0.025, 0.0, 0.08);
    float tooth2 = sdRoundCone(pTeeth - vec3(0.095, 0.01, -0.02), 0.025, 0.0, 0.07);
    float dTeeth = min(tooth1, tooth2);

    float dBodySphere = sdSphere(p, 0.35);
    
    vec3 pMouth = p - vec3(0.0, -0.12, 0.2); 
    float dMouthBlob = sdSphere(vec3(pMouth.x, pMouth.y*1.4, pMouth.z), 0.17);
    float flatTopPlane = p.y - (-0.08); 
    float dMouthCutShape = max(dMouthBlob, flatTopPlane);

    float dGreenBody = opSmoothSub(dMouthCutShape, dBodySphere, 0.02);

    float dMouthInner = dMouthCutShape + 0.01;
    
    dMouthInner = max(dMouthInner, -dTeeth + 0.02);

    vec3 pEyes = p - vec3(0.0, 0.25, 0.15);
    float dEyeL = sdSphere(pEyes - vec3(-0.13, 0.0, 0.0), 0.14);
    float dEyeR = sdSphere(pEyes - vec3(0.13, 0.0, 0.0), 0.14);
    float dEyes = min(dEyeL, dEyeR);
    
    float dPupilL = sdSphere(pEyes - vec3(-0.11, 0.02, 0.11), 0.04);
    float dPupilR = sdSphere(pEyes - vec3(0.11, 0.02, 0.11), 0.04);
    float dPupils = min(dPupilL, dPupilR);
    
    float legFL = sdSphere(p - vec3(-0.25, -0.3, 0.15), 0.1);
    float legFR = sdSphere(p - vec3(0.25, -0.3, 0.15), 0.1);
    float legBL = sdSphere(p - vec3(-0.25, -0.3, -0.15), 0.1);
    float legBR = sdSphere(p - vec3(0.25, -0.3, -0.15), 0.1);
    float dLegs = min(min(legFL, legFR), min(legBL, legBR));
    
    float dAntenna = sdCapsule(p, vec3(0.0, 0.35, 0.0), vec3(0.0, 0.55, 0.0), 0.02);
    float dAntennaBall = sdSphere(p - vec3(0.0, 0.55, 0.0), 0.04);
    
    float greenPart = smin(dGreenBody, dLegs, 0.05);
    greenPart = smin(greenPart, dAntenna, 0.02);
    greenPart = smin(greenPart, dAntennaBall, 0.02);
    vec4 res = vec4(greenPart, vec3(0.35, 0.85, 0.25)); 
    
    if (dMouthInner < res.x) {
        res = vec4(dMouthInner, vec3(0.15, 0.02, 0.02)); 
    }

    float whiteParts = dEyes;
    if (whiteParts < res.x) res = vec4(whiteParts, vec3(1.0));
    
    if (dPupils < res.x) res = vec4(dPupils, vec3(0.0));
    
    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 monster = sdMonster(p);
    vec4 omnom = sdOmNom(p);
    vec4 res = opUnion(monster, omnom);

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

    vec3 ray_origin = vec3(0.0, 0.3, 3.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh + vec2(0.0, 0.2), -2.0));
    vec4 res = raycast(ray_origin, ray_direction);



    vec3 col = res.yzw;


    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);

    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);

    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    bool isMouth = (col.r < 0.5 && col.r > col.g*2.0 && col.b < 0.1);
    if (isMouth) {
        shad *= 0.4;
    }    
    col *= shad;
    float specIntensity = isMouth ? 0.0 : 1.0;
    
    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec * specIntensity;

    fragColor = vec4(col, 1.0);
}