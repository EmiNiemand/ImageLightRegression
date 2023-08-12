#version 430
in vec3 uv;

out vec4 fragColor;

uniform samplerCube cubeMapTexture;

void main()
{
    fragColor = texture(cubeMapTexture, uv);
}