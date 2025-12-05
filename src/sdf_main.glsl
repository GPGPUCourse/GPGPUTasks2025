// Smooth minimum for organic blending
float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float sdEllipsoid( vec3 p, vec3 r )
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

// 2D Rotation matrix
mat2 rot2D(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

// Plane
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

// --- Animation Logic --
// Returns the current animation time within a loop
float getAnimTime() {
    // Loop the scene every 12 seconds
    return mod(iTime, 12.0);
}

vec4 sdBody(vec3 p)
{
    float t = getAnimTime();
    vec3 bodyColor = vec3(0.2, 0.8, 0.2); // Green
    
    // --- Body ---
    // Centered slightly up
    float d = sdEllipsoid(p - vec3(0.0, 0.4, 0.0), vec3(0.3, 0.38, 0.3));

    // --- Legs ---
    float legL = sdCapsule(p, vec3(-0.15, 0.2, 0.0), vec3(-0.2, 0.0, 0.0), 0.06);
    float legR = sdCapsule(p, vec3( 0.15, 0.2, 0.0), vec3( 0.2, 0.0, 0.0), 0.06);
    
    d = smin(d, legL, 0.08);
    d = smin(d, legR, 0.08);

    // --- Tail ---
    float tail = sdCapsule(p, vec3(0.0, 0.2, -0.2), vec3(0.0, 0.05, -0.30), 0.08);
    d = smin(d, tail, 0.1);

    // --- Arms Animation Logic ---
    
    // Phase 1: Wiggle (0s to 4s)
    // Phase 2: Transition to T-Pose (4s to 5s)
    // Phase 3: Hold T-Pose (5s+)
    
    float tPoseBlend = smoothstep(4.0, 5.0, t); // 0.0 = Normal/Wiggle, 1.0 = T-Pose
    float armLength = 0.25;
    float armRadius = 0.05;

    // -- Left Arm --
    vec3 L_Shoulder = vec3(-0.25, 0.45, 0.0);
    
    // Calculate Wiggle Position
    float wiggleAngleL = lazycos(t * 6.0 + 3.14159) * 0.6; // Phase offset by π
    vec2 wiggleDirL2D = rot2D(wiggleAngleL) * vec2(-0.6, 0.8); // Mirror X direction
    vec3 L_Hand_Wave = L_Shoulder + vec3(wiggleDirL2D.x, wiggleDirL2D.y, 0.1) * armLength;
    
    vec3 L_Hand_TPose = L_Shoulder + vec3(-armLength, 0.0, 0.0); // Sticking out X

    vec3 L_Hand_Current = mix(L_Hand_Wave, L_Hand_TPose, tPoseBlend);
    float armL = sdCapsule(p, L_Shoulder, L_Hand_Current, armRadius);
    d = smin(d, armL, 0.05);

    // -- Right Arm --
    vec3 R_Shoulder = vec3(0.25, 0.45, 0.0);
    vec3 R_Hand_Idle = vec3(0.45, 0.3, 0.1); // Hanging down
    vec3 R_Hand_TPose = R_Shoulder + vec3(armLength, 0.0, 0.0); // Sticking out X

    vec3 R_Hand_Current = mix(R_Hand_Idle, R_Hand_TPose, tPoseBlend);
    float armR = sdCapsule(p, R_Shoulder, R_Hand_Current, armRadius);
    
    d = smin(d, armR, 0.05);

    return vec4(d, bodyColor);
}

vec4 sdEye(vec3 p)
{
    vec3 eyeCenter = vec3(0.0, 0.5, 0.26); 
    float eyeRadius = 0.17;
    
    float d = length(p - eyeCenter) - eyeRadius;

    // Color Logic for the eye parts
    vec3 col = vec3(1.0); // Default Sclera (White)
    
    // Calculate vector from eye center to surface point
    vec3 localDir = normalize(p - eyeCenter);
    
    // Dot product with "forward" vector (0,0,1) determines how close we are to the front of the eye
    float lookDot = dot(localDir, vec3(0.0, 0.1, 0.95)); // Look slightly up
    
    if (lookDot > 0.90) {
        col = vec3(0.0); // Pupil (Black)
    } else if (lookDot > 0.80) {
        col = vec3(0.2, 0.9, 1.0); // Cornea (Blue)
    } else {
        col = vec3(1.0); // Sclera White
    }

    return vec4(d, col);
}

vec4 sdMonster(vec3 p)
{
    float t = getAnimTime();

    // --- Global Transform (Fly & Spin) ---
    // Start flying after becoming T-Pose (around 5.0s)
    float flyStart = 5.0;
    
    if (t > flyStart) {
        float flyTime = t - flyStart;
        
        // 1. Translation: Fly Up
        // Acceleration: 0.5 * a * t^2
        float height = 0.2 * flyTime * flyTime; 
        p.y -= height;
        
        // 2. Rotation: Spin
        float spinSpeed = 3.0 + flyTime * 2.0; // Spining gets faster
        p.xz *= rot2D(flyTime * spinSpeed);
    }

    // --- Calculate SDFs ---

    vec4 body = sdBody(p);
    vec4 eye = sdEye(p);
    
    vec4 res = body;
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
    const float eps = 0.0001;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
    sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
    sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}

vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    float EPS = 1e-3;
    float t = 0.0;

    for (int iter = 0; iter < 200; ++iter) {
        vec3 p = ray_origin + t*ray_direction;
        vec4 res = sdTotal(p);
        
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
        if (t > 40.0) break; 
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


    vec3 ray_origin = vec3(0.0, 0.5, 2.0);
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