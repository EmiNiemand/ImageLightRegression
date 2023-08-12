#version 430

layout (location = 0) in vec3 inPosition;

out vec3 uv;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    uv = inPosition;
    mat4 noTranslationView = view;
    // Remove translation from view matrix
    noTranslationView[3][0] = 0;
    noTranslationView[3][1] = 0;
    noTranslationView[3][2] = 0;
    vec4 outPosition = projection * noTranslationView * vec4(inPosition, 1.0);
    gl_Position = outPosition.xyww;
}
