#version 430
#define NR_DIRECTIONAL_LIGHTS 4
#define NR_POINT_LIGHTS 4
#define NR_SPOT_LIGHTS 4

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

out vec2 uv;
out vec3 normal;
out vec3 fragPosition;
out vec4 fragPositionDirectionalLightSpaces[NR_DIRECTIONAL_LIGHTS];
out vec4 fragPositionSpotLightSpaces[NR_SPOT_LIGHTS];

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 directionalLightSpaceMatrices[NR_DIRECTIONAL_LIGHTS];
uniform mat4 spotLightSpaceMatrices[NR_SPOT_LIGHTS];

uniform vec2 texStrech;

void main() {
    fragPosition = vec3(model * vec4(inPosition, 1.0));
    uv = vec2(inUV.x * texStrech.x, inUV.y * texStrech.y);
    normal = mat3(transpose(inverse(model))) * inNormal;
    for (int i = 0; i < NR_DIRECTIONAL_LIGHTS; ++i) {
        fragPositionDirectionalLightSpaces[i] = directionalLightSpaceMatrices[i] * vec4(fragPosition, 1.0);
    }
    for (int i = 0; i < NR_SPOT_LIGHTS; ++i) {
        fragPositionSpotLightSpaces[i] = spotLightSpaceMatrices[i] * vec4(fragPosition, 1.0);
    }
    gl_Position = projection * view * model * vec4(inPosition, 1.0);
}
