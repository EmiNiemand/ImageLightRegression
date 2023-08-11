#version 430
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube cubeMapTexture;

void main()
{
    FragColor = texture(cubeMapTexture, TexCoords);
}