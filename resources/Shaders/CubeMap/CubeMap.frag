#version 430
in vec3 uv;

layout (location = 0) out vec3 screenTexture;
layout (location = 1) out vec4 selectedObjectTexture;

uniform samplerCube cubeMapTexture;
uniform bool isSelected = false;

void main()
{
    screenTexture = vec3(texture(cubeMapTexture, uv));
    if (isSelected) {
        selectedObjectTexture = vec4(1.0f);
    }
}