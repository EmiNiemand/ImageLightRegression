#version 430

//layout (location = 0) out vec4 screenTexture;

in vec2 TexCoords;
in vec3 Normal;
in vec3 FragPos;
in vec4 FragPosLightSpace;

void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
