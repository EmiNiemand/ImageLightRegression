#version 430

layout (location = 0) in vec2 inPosition;
layout (location = 1) in vec2 inUV;

out vec2 uv;


void main()
{
    uv = inUV;

    gl_Position = vec4(inPosition.x, inPosition.y, 0.0, 1.0);
}
