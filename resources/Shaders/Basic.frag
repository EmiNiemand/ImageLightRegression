#version 430
out vec4 FragColor;
//layout (location = 0) out vec4 screenTexture;

// STRUCTS
// -------

struct DirectionalLight {
    bool isActive;
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 color;
};

struct PointLight {
    bool isActive;
    vec3 position;
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 color;
};

struct SpotLight {
    bool isActive;
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 color;
};

struct Material {
    vec3 color;
    float shininess;
    float reflection;
    float refraction;
};

// CONSTANTS
// ---------

#define NR_POINT_LIGHTS 16
#define NR_DIRECTIONAL_LIGHTS 16
#define NR_SPOT_LIGHTS 16

// SHADER PASSED VALUES
// --------------------

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;
in vec4 FragPosLightSpace;

// UNIFORMS
// --------

uniform sampler2D texture_diffuse;
uniform sampler2D texture_specular;

// EXTERNALLY SET VARIABLES
// ------------------------

uniform vec3 viewPos;
uniform DirectionalLight directionalLights[NR_DIRECTIONAL_LIGHTS];
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLights[NR_SPOT_LIGHTS];

uniform Material material = Material(vec3(1, 1, 1), 32.0f, 0.0f, 0.0f);

// FORWARD DECLARATIONS
// --------------------

vec3[3] CalculateDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir);
vec3[3] CalculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
vec3[3] CalculateSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir);
//float ShadowCalculation(vec4 fragPosLightSpace);

float shadow;
// MAIN
// ----

void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(viewPos - FragPos);

    vec3 result = vec3(0.0f);

    vec3[3] lightSettings;

//    shadow = ShadowCalculation(FragPosLightSpace);

    shadow = 0.0f;
    // phase 1: directional lights
    for(int i = 0; i < NR_DIRECTIONAL_LIGHTS; i++){
        if(directionalLights[i].isActive) {
            lightSettings = CalculateDirectionalLight(directionalLights[i], N, V);
            result += lightSettings[0] + (1 - shadow) * (lightSettings[1] + lightSettings[2]);
        }
    }

    // phase 2: point lights
    for(int i = 0; i < NR_POINT_LIGHTS; i++){
        if(pointLights[i].isActive){
            lightSettings = CalculatePointLight(pointLights[i], N, FragPos, V);
            result += lightSettings[0] + (1 - shadow) * (lightSettings[1] + lightSettings[2]);
        }
    }

    // phase 3: spot lights
    for(int i = 0; i < NR_SPOT_LIGHTS; i++){
        if(spotLights[i].isActive){
            lightSettings = CalculateSpotLight(spotLights[i], N, FragPos, V);
            result += lightSettings[0] + (1 - shadow) * (lightSettings[1] + lightSettings[2]);
        }
    }
    result = result * material.color;

//    if(material.reflection > 0.001) {
//        vec3 I = -V;
//        vec3 R = reflect(I, N);
//        result = mix(result, texture(skybox, R).rgb, material.reflection);
//    }
//
//    if(material.refraction > 0.001) {
//        // How much light bends
//        float ratio = 1.00 / 1.52;
//        vec3 I = -V;
//        vec3 R = refract(I, N, ratio);
//        result = mix(result, texture(skybox, R).rgb, material.refraction);
//    }

    //screenTexture = vec4(result, 1.0f);
    FragColor = vec4(result, 1.0f);
}


// LIGHT FUNCTIONS
// ---------------

vec3[3] CalculateDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir)
{
    vec3 lightDir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
    // combine results
    vec3 ambient = light.ambient * vec3(texture(texture_diffuse, TexCoords)) * light.color;
    vec3 diffuse = light.diffuse * diff * vec3(texture(texture_diffuse, TexCoords)) * light.color;
    vec3 specular = light.specular * spec * vec3(texture(texture_specular, TexCoords)) * light.color;
    vec3[3] lightSettings = {ambient, diffuse, specular};
    return lightSettings;
}


vec3[3] CalculatePointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
    // attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // combine results
    vec3 ambient = light.ambient * vec3(texture(texture_diffuse, TexCoords)) * light.color;
    vec3 diffuse = light.diffuse * diff * vec3(texture(texture_diffuse, TexCoords)) * light.color;
    vec3 specular = light.specular * spec * vec3(texture(texture_specular, TexCoords)) * light.color;
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    vec3[3] lightSettings = {ambient, diffuse, specular};
    return lightSettings;
}


vec3[3] CalculateSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir)
{
    vec3 lightDir = normalize(light.position - fragPos);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), material.shininess);
    // attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    // combine results
    vec3 ambient = light.ambient * vec3(texture(texture_diffuse, TexCoords)) * light.color;
    vec3 diffuse = light.diffuse * diff * vec3(texture(texture_diffuse, TexCoords)) * light.color;
    vec3 specular = light.specular * spec * vec3(texture(texture_specular, TexCoords)) * light.color;
    ambient *= attenuation * intensity;
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;
    vec3[3] lightSettings = {ambient, diffuse, specular};
    return lightSettings;
}

//float ShadowCalculation(vec4 fragPosLightSpace)
//{
//    // perform perspective divide
//    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
//    // transform to [0,1] range
//    projCoords = projCoords * 0.5 + 0.5;
//    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
//    float closestDepth = texture(shadowMap, projCoords.xy).r;
//    // get depth of current fragment from light's perspective
//    float currentDepth = projCoords.z;
//
//    // PCF
//    float shadow = 0.0;
//    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
//    for(int x = -1; x <= 1; ++x)
//    {
//        for(int y = -1; y <= 1; ++y)
//        {
//            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
//            shadow += currentDepth > pcfDepth  ? 1.0 : 0.0;
//        }
//    }
//    shadow /= 9.0;
//
//    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
//    if(projCoords.z > 1.0) shadow = 0.0;
//
//    return shadow;
//}