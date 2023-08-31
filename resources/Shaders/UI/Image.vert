#version 430
layout(location = 0) in vec2 inPosition;

out vec2 uv;

uniform vec2 size;
uniform vec2 screenPosition;
uniform vec2 pivot = vec2(0.5f, 0.5f);

void main() {
    vec2 outPosition = inPosition.xy;
    outPosition *= size * 2;
    outPosition += screenPosition;

    uv = inPosition.xy + pivot;

    gl_Position = vec4(outPosition, 0.0, 1.0);
}