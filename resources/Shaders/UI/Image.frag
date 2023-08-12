#version 430
in vec2 uv;

out vec4 fragColor;

uniform sampler2D imageTexture;

void main()
{
    fragColor = texture(imageTexture, uv);
}