#version 430
// CONSTANTS
// ---------

#define NR_DIRECTIONAL_LIGHTS 4
#define NR_SPOT_LIGHTS 4
#define NR_POINT_LIGHTS 4


// SHADER PASSED VALUES
// --------------------

in vec2 uv;
in vec3 normal;
in vec3 fragPosition;
in vec4 fragPositionDirectionalLightSpaces[NR_DIRECTIONAL_LIGHTS];
in vec4 fragPositionSpotLightSpaces[NR_SPOT_LIGHTS];

layout (location = 0) out vec3 screenTexture;
layout (location = 1) out vec4 selectedObjectTexture;

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

// UNIFORMS
// --------

uniform sampler2D textureDiffuse;
uniform sampler2D textureSpecular;
uniform sampler2D textureNormal;
uniform sampler2D textureHeight;

uniform samplerCube cubeMapTexture;
uniform sampler2D directionalLightShadowMapTexture[NR_DIRECTIONAL_LIGHTS];
uniform sampler2D spotLightShadowMapTexture[NR_SPOT_LIGHTS];
uniform samplerCube pointLightShadowMapTexture[NR_POINT_LIGHTS];

// EXTERNALLY SET VARIABLES
// ------------------------

uniform DirectionalLight directionalLights[NR_DIRECTIONAL_LIGHTS];
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight spotLights[NR_SPOT_LIGHTS];

uniform vec3 pointLightPositions[NR_POINT_LIGHTS];
uniform vec3 viewPosition;
uniform float farPlane;

uniform Material material = Material(vec3(1, 1, 1), 32.0f, 0.0f, 0.0f);

uniform bool isSelected;

// FORWARD DECLARATIONS
// --------------------

vec3[3] CalculateDirectionalLight(DirectionalLight light, vec3 inNormal, vec3 inViewDirection);
vec3[3] CalculatePointLight(PointLight light, vec3 inNormal, vec3 inFragPosition, vec3 inViewDirection);
vec3[3] CalculateSpotLight(SpotLight light, vec3 inNormal, vec3 inFragPosition, vec3 inViewDirection);
float CalculateDirectionalAndSpotLightShadow(vec4 inFragPositionLightSpace, sampler2D shadowMapTexture);
float CalculatePointLightShadow(vec3 lightPosition, samplerCube shadowMapTexture);


// Variables
// --------------------

// array of offset direction for sampling
vec3 gridSamplingDisk[20] = vec3[]
(
vec3(1, 1,  1), vec3( 1, -1,  1), vec3(-1, -1,  1), vec3(-1, 1,  1),
vec3(1, 1, -1), vec3( 1, -1, -1), vec3(-1, -1, -1), vec3(-1, 1, -1),
vec3(1, 1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0), vec3(-1, 1,  0),
vec3(1, 0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1, 0, -1),
vec3(0, 1,  1), vec3( 0, -1,  1), vec3( 0, -1, -1), vec3( 0, 1, -1)
);

// MAIN
// ----

void main() {
    vec3 normalizedNormal = normalize(normal);
    vec3 viewDirection = normalize(viewPosition - fragPosition);

    vec3 result = vec3(0.0f);

    vec3[3] lightSettings;
    float shadow = 0.0f;

    // phase 1: directional lights
    for(int i = 0; i < NR_DIRECTIONAL_LIGHTS; i++) {
        shadow = 0.0f;
        if(directionalLights[i].isActive) {
            shadow = CalculateDirectionalAndSpotLightShadow(fragPositionDirectionalLightSpaces[i], directionalLightShadowMapTexture[i]);

            lightSettings = CalculateDirectionalLight(directionalLights[i], normalizedNormal, viewDirection);
            result += lightSettings[0] + (1 - shadow) * (lightSettings[1] + lightSettings[2]);
        }
    }

    // phase 2: point lights
    for(int i = 0; i < NR_POINT_LIGHTS; i++) {
        shadow = 0.0f;

        if(pointLights[i].isActive) {
            shadow = CalculatePointLightShadow(pointLightPositions[i], pointLightShadowMapTexture[i]);

            lightSettings = CalculatePointLight(pointLights[i], normalizedNormal, fragPosition, viewDirection);
            result += lightSettings[0] + (1 - shadow) * (lightSettings[1] + lightSettings[2]);
        }
    }

    // phase 3: spot lights
    for(int i = 0; i < NR_SPOT_LIGHTS; i++) {
        shadow = 0.0f;

        if(spotLights[i].isActive) {
            shadow = CalculateDirectionalAndSpotLightShadow(fragPositionSpotLightSpaces[i], spotLightShadowMapTexture[i]);

            lightSettings = CalculateSpotLight(spotLights[i], normalizedNormal, fragPosition, viewDirection);
            result += lightSettings[0] + (1 - shadow) * (lightSettings[1] + lightSettings[2]);
        }
    }
    result = result * material.color;

    if(material.reflection > 0.001) {
        vec3 negatedViewDirection = -viewDirection;
        vec3 reflectUV = reflect(negatedViewDirection, normalizedNormal);
        result = mix(result, texture(cubeMapTexture, reflectUV).rgb, material.reflection);
    }

    if(material.refraction > 0.001) {
        // How much light bends
        float ratio = 1.00 / 1.52;
        vec3 negatedViewDirection = -viewDirection;
        vec3 refractUV = refract(negatedViewDirection, normalizedNormal, ratio);
        result = mix(result, texture(cubeMapTexture, refractUV).rgb, material.refraction);
    }

    screenTexture = result;
    if (isSelected) {
        selectedObjectTexture = vec4(1.0f);
    }
}


// LIGHT FUNCTIONS
// ---------------

vec3[3] CalculateDirectionalLight(DirectionalLight light, vec3 inNormal, vec3 inViewDirection)
{
    vec3 lightDirection = normalize(-light.direction);
    // diffuse shading
    float diffuseShading = max(dot(inNormal, lightDirection), 0.0);
    // specular shading
    vec3 halfwayDirection = normalize(lightDirection + inViewDirection);
    float specularShading = pow(max(dot(inNormal, halfwayDirection), 0.0), material.shininess);
    // combine results
    vec3 ambient = light.ambient * vec3(texture(textureDiffuse, uv)) * light.color;
    vec3 diffuse = light.diffuse * diffuseShading * vec3(texture(textureDiffuse, uv)) * light.color;
    vec3 specular = light.specular * specularShading * vec3(texture(textureSpecular, uv)) * light.color;
    vec3[3] lightSettings = {ambient, diffuse, specular};
    return lightSettings;
}


vec3[3] CalculatePointLight(PointLight light, vec3 inNormal, vec3 inFragPosition, vec3 inViewDirection)
{
    vec3 lightDirection = normalize(light.position - inFragPosition);
    // diffuse shading
    float diffuseShading = max(dot(inNormal, lightDirection), 0.0);
    // specular shading
    vec3 halfwayDirection = normalize(lightDirection + inViewDirection);
    float specularShading = pow(max(dot(inNormal, halfwayDirection), 0.0), material.shininess);
    // attenuation
    float distance = length(light.position - inFragPosition);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // combine results
    vec3 ambient = light.ambient * vec3(texture(textureDiffuse, uv)) * light.color;
    vec3 diffuse = light.diffuse * diffuseShading * vec3(texture(textureDiffuse, uv)) * light.color;
    vec3 specular = light.specular * specularShading * vec3(texture(textureSpecular, uv)) * light.color;
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    vec3[3] lightSettings = {ambient, diffuse, specular};
    return lightSettings;
}


vec3[3] CalculateSpotLight(SpotLight light, vec3 inNormal, vec3 inFragPosition, vec3 inViewDirection)
{
    vec3 lightDirection = normalize(light.position - inFragPosition);
    // diffuse shading
    float diffuseShading = max(dot(inNormal, lightDirection), 0.0);
    // specular shading
    vec3 halfwayDirection = normalize(lightDirection + inViewDirection);
    float specularShading = pow(max(dot(inNormal, halfwayDirection), 0.0), material.shininess);
    // attenuation
    float distance = length(light.position - inFragPosition);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // spotlight intensity
    float theta = dot(lightDirection, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    // combine results
    vec3 ambient = light.ambient * vec3(texture(textureDiffuse, uv)) * light.color;
    vec3 diffuse = light.diffuse * diffuseShading * vec3(texture(textureDiffuse, uv)) * light.color;
    vec3 specular = light.specular * specularShading * vec3(texture(textureSpecular, uv)) * light.color;
    ambient *= attenuation * intensity;
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;
    vec3[3] lightSettings = {ambient, diffuse, specular};
    return lightSettings;
}

float CalculateDirectionalAndSpotLightShadow(vec4 inFragPositionLightSpace, sampler2D shadowMapTexture)
{
    // perform perspective divide
    vec3 projectionUV = inFragPositionLightSpace.xyz / inFragPositionLightSpace.w;
    // transform to [0,1] range
    projectionUV = projectionUV * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMapTexture, projectionUV.xy).r;
    // get depth of current fragment from light's perspective
    float currentDepth = projectionUV.z;

    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMapTexture, 0);
    for(int x = -1; x <= 1; ++x)
    {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMapTexture, projectionUV.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth > pcfDepth  ? 1.0 : 0.0;
        }
    }
    shadow /= 9.0;

    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(projectionUV.z > 1.0) shadow = 0.0;
    // keep shadow scale if material is transparent
    if(material.refraction > 0.0f) shadow *= (1 - material.refraction);

    return shadow;
}

float CalculatePointLightShadow(vec3 lightPosition, samplerCube shadowMapTexture)
{
    vec3 fragToLight = fragPosition - lightPosition;

    float currentDepth = length(fragToLight);

    float shadow = 0.0;
    int samples = 20;
    float viewDistance = length(viewPosition - fragPosition);
    float diskRadius = (1.0 + (viewDistance / farPlane)) / 25;
    for(int i = 0; i < samples; ++i)
    {
        float closestDepth = texture(shadowMapTexture, fragToLight + gridSamplingDisk[i] * diskRadius).r;
        closestDepth *= farPlane;   // undo mapping [0;1]
        if(currentDepth > closestDepth)
            shadow += 1.0;
    }
    shadow /= float(samples);

    return shadow;
}