#version 430
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

out vec2 uv;
out vec3 normal;
out vec3 fragPosition;
out vec4 fragPositionLightSpace;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 lightSpaceMatrix = mat4(0.0f);

uniform vec2 texStrech;

void main() {
    fragPosition = vec3(model * vec4(inPosition, 1.0));
    uv = vec2(inUV.x * texStrech.x, inUV.y * texStrech.y);
    normal = mat3(transpose(inverse(model))) * inNormal;
    fragPositionLightSpace = lightSpaceMatrix * vec4(fragPosition, 1.0);
    gl_Position = projection * view * model * vec4(inPosition, 1.0);
}
