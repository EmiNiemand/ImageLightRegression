#version 430
layout(location = 0) in vec3 inPosition;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

uniform int isSkybox = 0;

void main() {
    vec4 outPosition;
    if (isSkybox == 1) {
        mat4 noTranslationView = view;
        // Remove translation from view matrix
        noTranslationView[3][0] = 0;
        noTranslationView[3][1] = 0;
        noTranslationView[3][2] = 0;
        outPosition = vec4(projection * noTranslationView * vec4(inPosition, 1.0)).xyww;
    }
    if (isSkybox == 0) {
        outPosition = projection * view * model * vec4(inPosition, 1.0);
    }

    gl_Position = outPosition;
}
