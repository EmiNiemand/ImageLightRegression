#version 430
in vec3 uv;

layout (location = 0) out vec3 screenTexture;

uniform samplerCube cubeMapTexture;

void main()
{
    screenTexture = vec3(texture(cubeMapTexture, uv));
}